import os
import datetime

from flask import Flask, render_template,request

from google.cloud import storage
import google.auth

import secrets_1
from rsc.llm_main import SummarizationSession

# from google.oauth2 import service_account


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  

        # Get the file input.
        f = request.files['file']

        # Get the values of the checkbox inputs.
        data_cleaning = request.form.get('data_cleaning')
        prompt_structuring = request.form.get('prompt_structuring')
        iterative_prompting = request.form.get('iterative_prompting')

        print(data_cleaning)
        print(prompt_structuring)
        print(iterative_prompting)

        ts = int(datetime.datetime.now().timestamp() * 1000)

        credentials, project_id = google.auth.load_credentials_from_file(secrets_1.gcp_credential_file)

        bucket = storage.Client(project=project_id, credentials=credentials).bucket('raw_transcripts_bucket')
        blob = bucket.blob(f'{ts}_input.txt')
        blob.upload_from_file(f)

        session = SummarizationSession(file_blob=blob, timestamp=ts)
        sum_result = session()

        return render_template("acknowledgement.html", summary = sum_result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))