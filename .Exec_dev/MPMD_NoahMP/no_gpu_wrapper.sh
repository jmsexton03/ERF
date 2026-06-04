#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
echo "CUDA is $CUDA_VISIBLE_DEVICES"
exec "$@"
