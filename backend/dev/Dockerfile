FROM jules-backend:latest

WORKDIR /app

# Install Jupyter and papermill for notebook execution/testing
RUN pip install --no-cache-dir jupyterlab papermill

# Optionally copy notebooks into the image (or mount as a volume in dev)
COPY ../../ipy /app/ipy

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token="]
