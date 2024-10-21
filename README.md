# AutoSync-Video-Transcription-and-Audio-Correction-Tool
A streamlit based app which corrects grammatical mistakes from your video and replaces it with grammatically correct synthesized audio

## How to use
- Clone the repository
- Install the requirements using `pip install -r requirements.txt`
- Setup a secrets.toml file and place it in `Users\%username%\.streamlit` with the following contents
```
["AZURE_CREDENTIALS"]
AZURE_OPENAI_ENDPOINT="<YOUR_AZURE_API_ENDPOINT>"
AZURE_OPENAI_API_KEY="<YOUR_AZURE_OPENAI_API_KEY>"

["GOOGLE_CREDENTIALS"]
creds = <GOOGLE_SERVICE_ACCOUNT_CREDENTIALS>
```
- Run the app using `streamlit run app.py`

## Features
- Makes use of GPT-4o to fix your grammatical mistakes
- Ensures that the corrected transcript remains true to the original transcript as much as possible
- Syncs the synthesized audio as close as possible to the original audio, with ensuring the rate of speech using the original audio
- Provides a frontend to upload your video and outputs a video with corrected audio
- Makes sure that common words are aligned properly with the timestamps from the original video such that lip sync is ensured wherever possible
- Allows the user to change the gender of the voice

![image](https://github.com/user-attachments/assets/61debd13-8922-4422-a5ac-c41295b46d8e)
![image](https://github.com/user-attachments/assets/d4109f4e-f90f-442b-9bca-a9994ad7d3a6)
