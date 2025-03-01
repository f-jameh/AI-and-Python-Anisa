#!/usr/bin/env python3
from app import app

if __name__ == "__main__":
    app.run(host="172.16.1.1", port=5002, debug=False, use_reloader=False)
