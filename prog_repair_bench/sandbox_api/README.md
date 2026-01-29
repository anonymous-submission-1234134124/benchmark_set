
Edit last line of:

`prog_repair_bench/prog_repair_bench/sandbox_api/Dockerfile`

`CMD ["python", "prog_repair_bench/sandbox_api/run_api.py", "--num_servers", "4", "--repo", "sympy"]`

You can change num servers and repo. 

Build an image:

`python prog_repair_bench/sandbox_api/build_sandbox_api_image.py` 

Run the service:

`kubectl apply -f prog_repair_bench/sandbox_api/sandbox-api.yaml` 

Delete the service:

`kubectl delete deployment sandbox-sympy-api`
`kubectl delete service sandbox-sympy-api`