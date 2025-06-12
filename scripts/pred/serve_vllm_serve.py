# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/vllm-project/vllm/blob/v0.4.0/vllm/entrypoints/api_server.py

import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import aiohttp

from vllm.engine.arg_utils import AsyncEngineArgs

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
vllm_server_url = "http://localhost:8094"  # Will be set via command line argument


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)



@app.put("/generate")
async def completions(request: Request) -> Response:
    """Generate completion by forwarding to an external vLLM server.
    
    This endpoint wraps around a vLLM server endpoint and returns responses
    in the same format as the /generate endpoint for backwards compatibility.
    
    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters compatible with vLLM completions API.
    """
    
    if vllm_server_url is None:
        return JSONResponse(
            {"error": "vLLM server URL not configured. Use --vllm-server-url argument."},
            status_code=500
        )
    
    request_dict = await request.json()
    request_dict["model"] = model_name
    print(f"Received request: {json.dumps(request_dict, indent=2)}")
    stream = request_dict.get("stream", False)
    
    # Forward the request to the vLLM server
    async with aiohttp.ClientSession() as session:
        try:
            vllm_completions_url = f"{vllm_server_url.rstrip('/')}/v1/completions"
            
            # Streaming case
            if stream:
                async def stream_vllm_results() -> AsyncGenerator[bytes, None]:
                    async with session.post(
                        vllm_completions_url,
                        json=request_dict,
                        headers={"Content-Type": "application/json"}
                    ) as vllm_response:
                        if vllm_response.status != 200:
                            error_text = await vllm_response.text()
                            yield (json.dumps({"error": f"vLLM server error: {error_text}"}) + "\0").encode("utf-8")
                            return
                        
                        async for chunk in vllm_response.content.iter_chunked(1024):
                            if chunk:
                                # Parse the vLLM streaming response and convert to our format
                                try:
                                    lines = chunk.decode('utf-8').strip().split('\n')
                                    for line in lines:
                                        if line.startswith('data: ') and not line.startswith('data: [DONE]'):
                                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                                            if 'choices' in data and len(data['choices']) > 0:
                                                text_outputs = [choice['text'] for choice in data['choices']]
                                                ret = {"text": text_outputs}
                                                yield (json.dumps(ret) + "\0").encode("utf-8")
                                except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
                                    # Skip malformed chunks
                                    continue
                
                return StreamingResponse(stream_vllm_results())
            
            # Non-streaming case
            else:
                async with session.post(
                    vllm_completions_url,
                    json=request_dict,
                    headers={"Content-Type": "application/json"}
                ) as vllm_response:
                    if vllm_response.status != 200:
                        error_text = await vllm_response.text()
                        return JSONResponse(
                            {"error": f"vLLM server error: {error_text}"},
                            status_code=vllm_response.status
                        )
                    
                    vllm_result = await vllm_response.json()
                    
                    # Convert vLLM response format to our format
                    if 'choices' in vllm_result and len(vllm_result['choices']) > 0:
                        text_outputs = [choice['text'] for choice in vllm_result['choices']]
                        ret = {"text": text_outputs}
                        return JSONResponse(ret)
                    else:
                        return JSONResponse({"error": "Invalid response from vLLM server"}, status_code=500)
        
        except aiohttp.ClientError as e:
            return JSONResponse(
                {"error": f"Failed to connect to vLLM server: {str(e)}"},
                status_code=503
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"Unexpected error: {str(e)}"},
                status_code=500
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8094",
        help="URL of the external vLLM server for /v1/completions endpoint (e.g., http://localhost:8000)")
    
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    model_name = args.model
    
    # Set the global vllm_server_url
    vllm_server_url = args.vllm_server_url

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)