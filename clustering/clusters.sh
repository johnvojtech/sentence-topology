#!/bin/bash
ls $2 | xargs -I % python3 clustering.py $1/% $2
