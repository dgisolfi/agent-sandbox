# agent-sandbox
Autonomous LLM agent that runs experiments inside an isolated sandbox container.

## Author(s)

- Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>

## Usage

```
docker compose run --rm agent list
```

### CPU Only

```
EXPERIMENT_ID=fibonacci docker-compose run --rm agent start fibonacci
```

Example Output
```
$ docker-compose run --rm agent start fibonacci
time="2026-05-18T20:42:51-04:00" level=warning msg="Found orphan containers ([agent-sandbox-agent-run-2972183af22a agent-sandbox-agent-run-c8a3ce2d0b0c]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up."
[+]  3/3t 3/32
 ✔ Container ollama     Healthy                                                                                                                                                                                                 0.5s
 ✔ Container sandbox    Running                                                                                                                                                                                                 0.0s
 ✔ Container model-init Started                                                                                                                                                                                                 0.2s
Container ollama Waiting 
Container sandbox Waiting 
Container model-init Waiting 
Container model-init Exited 
Container ollama Healthy 
Container sandbox Healthy 
Container agent-sandbox-agent-run-30664b8b4498 Creating 
Container agent-sandbox-agent-run-30664b8b4498 Created 
[Agent] Checking sandbox connection...
[Agent] Checking LLM connection...
[00:42:52] [Agent] Iteration 1/50
[00:43:48] [Plan] run_command (55.2s): pytest -q
[00:43:48] [Execute] ok=False exit=127 timed_out=False (0.0s)
[00:44:58] [Evaluate] continue (70.7s): The pytest command was not found, indicating that it is either not installed or not in the system's PATH. This prevents us from running the tests to understand the task and choose the first concrete development step.
[00:44:58] [Agent] Iteration 2/50
[00:46:12] [Plan] run_command (73.7s): pip install pytest
[00:46:14] [Execute] ok=True exit=0 timed_out=False (1.6s)

```


### GPU with local Ollama model


```
EXPERIMENT_ID=fibonacci docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm agent start fibonacci
```