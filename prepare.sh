#!/bin/bash
pip install -r requirements.txt \
            --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple
