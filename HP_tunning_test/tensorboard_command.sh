#!/bin/bash
echo `tensorboard --logdir=tensorboard/ --host 164.125.34.220 --port 6006 --reload_multifile=true`
#--host 를 안해주면 외부 접속이 안된다.
#--port는 자유
#--reload_multifile을 안해주면 새로고침이 안되는 버그가 있다.
