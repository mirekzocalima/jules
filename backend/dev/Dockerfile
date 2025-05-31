FROM jules-backend:latest

WORKDIR /app

# Install Jupyter and papermill for notebook execution/testing
RUN pip install --no-cache-dir jupyterlab papermill

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token="]
