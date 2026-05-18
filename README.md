# agent-sandbox
Autonomous LLM agent that runs experiments inside an isolated sandbox container.

## Author(s)

- Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>

## Usage


### CPU Only

```
docker compose run --rm agent start fibonacci
```


### GPU with local Ollama model


```
docker compose run -f docker-compose.gpu.yml --rm agent start fibonacci
```