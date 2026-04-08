ocker restart webtool_3-celery-worker-1   # stop a job 
● The issue is clear — your frontend container is stale (running for 25
  hours) while the celery-worker was only rebuilt 26 minutes ago. The        
  modified frontend files (build.py, check_results.py, inference.py) are not 
  in the running container.
                                                                             
  You also have modified backend files (main.py, tasks.py, model_build/), so 
  both need rebuilding.
                                                                             
  Run:            

  docker compose build --no-cache frontend backend celery-worker && docker
  compose up -d frontend backend celery-worker

  Or if you want to rebuild everything cleanly:                              
  
  docker compose up --build -d      

  changing from branch      