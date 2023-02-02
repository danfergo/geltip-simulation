import os
import time
from threading import Thread
import re

from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler

from ....experimenter import e
import json


def replace_all(x, config):
    # not very pretty code, but it works
    # replaces all ${key} -> config[key] in the attr.

    wrap = {'txt': x, 'offset': 0}
    regex = "(\$\{(\w(\w|\\n)*)\})"

    def replace(match, o):
        txt = o['txt']
        offset = o['offset']
        k = match.group(2)
        s = match.span(2)
        if k not in config:
            raise Exception('Config "' + k + '" not found !')

        o['txt'] = txt[: offset + s[0] - 2] + config[k] + txt[offset + s[1] + 1:]
        # offset is used to compensate the "original" spans
        # for the differences in the string before and after
        o['offset'] += len(config[k]) - (s[1] - s[0]) - 3

    ''
    [replace(x, wrap) for x in re.finditer(regex, x)]
    return wrap['txt']


def HandlerFactory(data_):
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.routes = {
                '/index.html': self.index,
                '/api': self.api
            }
            self.data = data_

            super().__init__(*args, directory=e.out(), **kwargs)

        # def __call__(self, *args, **kwargs):
        #     """Handle a request."""
        #     super().__init__(*args, directory=e.out(), **kwargs)

        def html(self, path, data=None):
            full_path = os.path.dirname(__file__) + '/html/' + path
            f = open(full_path, 'r')

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = replace_all(f.read(), data)
            self.wfile.write(bytes(html, "utf-8"))

        def json(self, data=None):
            data = data or None
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(data), 'utf-8'))

        def api(self):
            self.json(self.data)

        def index(self):

            card_template = """
                <div class="card">
                  <div class="card-body">
                    <h5 class="card-title">{src}</h5>
                    <img src="/{src}">
                  </div>
                </div>
            """

            plots = os.listdir(e.out())
            data = {
                'main_content': "\n".join([card_template.format(src=p) for p in plots])
            }
            self.html('/index.html', data)

        def do_GET(self):
            path = self.path
            path = '/index.html' if path == '/' else path

            if path in self.routes:
                self.routes[path]()
            else:
                super().do_GET()

        def log_message(self, format, *args):
            return

    return CustomHandler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EBoard:

    def __init__(self):
        hostName = "0.0.0.0"
        serverPort = 8080
        self.server = None
        self.data = {}
        self.thread = Thread(target=self.serve_on_port, args=[
            hostName,
            serverPort,
            self.data
        ])
        self.thread.start()

    def serve_on_port(self, hostname, port, data):
        handler = HandlerFactory(data)
        self.server = ThreadingHTTPServer((hostname, port), handler)
        print("Server started http://%s:%s" % (hostname, port))
        self.server.serve_forever()

    def on_epoch_end(self, ev):
        def assign(k, v):
            self.data[k] = v

        assign('epoch', ev['epoch'])

    def on_e_end(self, ev):
        self.server.shutdown()
        self.thread.join()
