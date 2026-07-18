from aiohttp import web


async def hello(request: web.Request) -> web.Response:
    return web.Response(text="VisionEdge Backend Running")


async def health(request: web.Request) -> web.Response:
    return web.json_response({"message": "VisionEdge Backend Running"})


@web.middleware
async def cors_middleware(request: web.Request, handler):
    response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    return response


app = web.Application(middlewares=[cors_middleware])
app.router.add_get("/", hello)
app.router.add_get("/health", health)


if __name__ == "__main__":
    web.run_app(app, host="127.0.0.1", port=8080)
