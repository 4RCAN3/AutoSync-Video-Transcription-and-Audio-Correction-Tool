#importing libraries
from moviepy.editor import VideoFileClip, AudioFileClip
from openai import AzureOpenAI
import os
import json
import pyttsx3
from pydub import AudioSegment
import re
from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
import shutil
import streamlit as st

class Word:
    def __init__(self, value: str, start_time, end_time) -> None:
        """Word class to store individual words and their start and end times

        Args:
            value (str): Value for the word
            start_time (float/int): Start time of the word in seconds
            end_time (float/int): End time of the word in seconds
        """        
        self.word = value
        self.start = start_time
        self.end = end_time
    
    def data(self) -> tuple:
        """Returns all the attributes of the word

        Returns:
            tuple: (Word, Start time, end time)
        """        
        return (self.word, self.start, self.end)
    

class AIaudioGen:
    def __init__(self, video_path: str, path_to_credentials: str, gender: str) -> None:
        """Fixes grammatical mistakes and synthesizes a new audio and syncs it with the original audio

        Args:
            video_path (str): path to the original video
            path_to_credentials (str): path to the google cloud credentials
            is_female (str): Gender of the voice
        """     

        self.video = VideoFileClip(video_path)
        self.og_audio = self.video.audio

        google_creds = st.secrets['GOOGLE_CREDENTIALS']['creds']
        credentials = service_account.Credentials.from_service_account_info(google_creds)
        self.stt_client = speech.SpeechClient(credentials=credentials)
        self.voice_engine = pyttsx3.init('sapi5')

        if gender == 'Female':
            #sets the voice to female
            voices = self.voice_engine.getProperty('voices')
            self.voice_engine.setProperty('voice', voices[1].id)

        self._create_openai_client()
        
        #Clear any old output files
        self.delete_directory('Output')
        
        os.makedirs('Output', exist_ok=True)
    

    def _create_openai_client(self) -> None:
        """
        Creates the opeanAI client and sets the prompt for it
        """        
        client = AzureOpenAI(
        azure_endpoint = st.secrets['AZURE_CREDENTIALS']['AZURE_OPENAI_ENDPOINT'], 
        api_key= st.secrets['AZURE_CREDENTIALS']["AZURE_OPENAI_API_KEY"],  
        api_version="2024-09-01-preview"
        )
        
        prompt = '''You need to fix any of the grammatical mistakes that are given to you by the user. 
        Remove any unprofessional language and words. 
        Write the message such that it sounds professional, but does not change the meaning or structure of the sentence.
        Do not change the wording of the original text.
        You do not need to add anything or change the meaning of the sentence.'''

        self.gpt_client, self.prompt = client, prompt
    
    def _remove_punctuation(self, text: str) -> str:
        """Remove specific punctuations from the text

        Args:
            text (str): Original string

        Returns:
            str: Cleaned string
        """        

        return re.sub(r"[,\./_=+\!@#$%^&*();]", '', text)
    
    def extract_audio(self) -> None:
        """
        Extracts audio from the original video
        """        

        self.og_audio.write_audiofile('Temp/og_audio.wav')
    
    def transcribe_audio(self) -> tuple:
        """Transcribes the audio from the original video using the google cloud speech to text API

        Returns:
            tuple: Returns the original transcripts and a list of "Word" objects containing timestamp information
        """        

        with open('Temp/og_audio.wav', "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,  
            language_code="en-US",
            audio_channel_count = 2,
            enable_word_time_offsets=True,
        )

        response = self.stt_client.recognize(config=config, audio=audio)

        transcript = []
        timestamp_info = []
        for result in response.results:
            alternative = result.alternatives[0]
            transcript.append(alternative.transcript)
            for word_info in alternative.words:
                word = word_info.word
                start_time = word_info.start_time
                end_time = word_info.end_time
                timestamp_info.append(Word(word, start_time.total_seconds(), end_time.total_seconds()))
        
        return ' '.join(transcript), timestamp_info
    

    def get_response(self, text: str) -> str:
        """Fetches responses from the openAI api using the GPT-4o model, that fixes the grammatical mistakes

        Args:
            text (str): Original transcript

        Returns:
            str: Corrected transcript
        """        

        response = self.gpt_client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": text},
            ]
            ,

        )
        
        return (response.model_dump_json(indent=2))
    

    def align_transcripts(self, original_words: list, timestamps: list[Word], response: list, window: int = 3) -> list[Word]:
        """Aligns the words from the corrected transcript to the words of the original transcript

        Args:
            original_words (list): List of words from the original transcript
            timestamps (list[Word]): List of Word objects with timestamp information from the original transcript
            response (list): List of words from the corrected transcript
            window (int, optional): A sliding window to move across the original list of words to match words 
            to the corrected transcript. Defaults to 3.

        Returns:
            list[Word]: List of Word objects with timestamp information for the corrected transcript
        """         

        corrected_timed_words = []
        non_occuring_words = []
        curr = 0 
        start = 0
        end = timestamps[-1].end

        for word in response:
            try:
                #Finds matching words from the original words in a sliding window
                word_idx = curr + original_words[curr: window+curr].index(word)
                og_word = timestamps[word_idx]

                #Concatenate words that do not occur in the original transcript and fix them according to the already set words
                if non_occuring_words != []:
                    prev_words = ' '.join(non_occuring_words)
                    non_occuring_words = []
                    corrected_timed_words.append(Word(value=prev_words,
                        start_time=start,
                        end_time=og_word.start,
                    ))
                
                corrected_timed_words.append(og_word)
                start = og_word.start
                curr = word_idx + 1

            #Word not found
            except ValueError:
                non_occuring_words.append(word)
        
        if non_occuring_words != []:
            prev_words = ' '.join(non_occuring_words)
            non_occuring_words = []
            corrected_timed_words.append(Word(
                value = prev_words,
                start_time=corrected_timed_words[-1].end,
                end_time=end,
            ))
        

        return corrected_timed_words
    
    def concatenate_simultaneous_words(self, words_with_timestamps: list[Word]) -> list[Word]:
        """Concatenate words that are spoken simultaneously 
        i.e conccurent words which have a matching end time and start time respectively

        Args:
            words_with_timestamps (list[Word]): List of Word objects with timestamp information for the corrected transcript

        Returns:
            list[Word]: Concatenated list of Word objects with timestamp information for the corrected transcript
        """        

        concatenated_words = []
        current_word = words_with_timestamps[0].word
        current_start = words_with_timestamps[0].start
        current_end = words_with_timestamps[0].end

        for i in range(1, len(words_with_timestamps)):
            word, start, end = words_with_timestamps[i].data()
            
            # If the start of the current word is the same as the end of the previous one, concatenate
            if start == current_end:
                current_word += " " + word
                current_end = end  # Update end time to the current word's end time
            else:
                # Otherwise, finalize the previous word and start a new one
                concatenated_words.append(Word(current_word, current_start, current_end))
                current_word = word
                current_start = start
                current_end = end
        
        # Append the last concatenated word
        concatenated_words.append(Word(current_word, current_start, current_end))
        
        return concatenated_words
    
    def save_word_audio(self, word: str, duration, filename: str, base_rate: int, base_duration):
        """Synthesize audio for a single word and save it to a file.

        Args:
            word (str): The word/words to synthesize audio for
            duration (int/float): Duration for which the word should be spoken for
            filename (str): Name of file where the audio will be saved
            base_rate (int): The base rate of words per minute at which the original video is spoken at
            base_duration (int/float): Mean duration for which a word is spoken at in the original video.

        Returns:
            Synthesized word/words according to the duration
        """        
        
        duration = duration if duration > 0.1 else 0.1
        rate_factor = duration / base_duration
        wpm = base_rate + base_rate*rate_factor if rate_factor < 1 else base_rate - base_rate*rate_factor
        self.voice_engine.setProperty('rate', wpm)
        
        self.voice_engine.save_to_file(word, filename)

        self.voice_engine.runAndWait()
        synthesized = AudioSegment.from_wav(filename)

        # Adjust speed according to duration (stretch or compress)
        speed_factor = len(synthesized) / (duration * 1000)  # Convert duration to milliseconds
        new_frame_rate = int(synthesized.frame_rate * speed_factor)
        modified_audio = synthesized._spawn(synthesized.raw_data, overrides={'frame_rate': new_frame_rate})

        return modified_audio.set_frame_rate(synthesized.frame_rate)

    def create_silence(self, duration_ms):
        """Create a silent audio segment for a given duration (in milliseconds).

        Args:
            duration_ms (int/float): The duration (in millisecond) for which the silent audio needs to be created

        Returns:
            A silent audio for x duration
        """      

        return AudioSegment.silent(duration=duration_ms)
    

    def synthesize_audio(self, words_with_timestamps: list[Word], output_filename: str, num_words: int):
        """Synthesizes audio for the corrected transcript

        Args:
            words_with_timestamps (list[Word]): A list of Word objects with timestamp information for the corrected transcript
            output_filename (str): The name of the file where the audio will be stored once synthesized
            num_words (int): Number of words in the original audio
        """        
        
        final_audio = AudioSegment.silent(0)  # Start with a blank segment
        prev_end_time = 0
        total_duration = words_with_timestamps[-1].end
        base_rate = (num_words/total_duration)*60
        base_duration = total_duration/num_words

        for word in words_with_timestamps:
            word, start, end = word.data()

            # If there's a gap between previous end time and the current start time, add silence
            if start > prev_end_time:
                silence_duration = (start - prev_end_time) * 1000  # Convert to milliseconds
                silence = self.create_silence(silence_duration)
                final_audio += silence

            # Save the word audio to a temporary file
            word_audio_file = f'Temp/{word}.wav'
            word_audio = self.save_word_audio(word, (end - start), word_audio_file, base_rate, base_duration)

            # Add the word audio to the final audio segment
            final_audio += word_audio

            # Update the previous end time
            prev_end_time = end

        # Export the final combined audio
        final_audio.export(output_filename, format='wav')

    def replace_audio(self, audio_path: str, output_path: str) -> None:
        """Replaces the audio with a synthesized audio from the original video

        Args:
            audio_path (str): Syntheized audio path
            output_path (str): Path to which the output video will be saved
        """        
        new_audio = AudioFileClip(audio_path)
        video = self.video.set_audio(new_audio)
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    def delete_directory(self, dir_name: str):
        """Delete any directory

        Args:
            dir_name (str): Name of the directory
        """        

        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    
    def run(self, output_audio_filename='output_audio.wav') -> tuple:
        """Runs all the proccess needed to transform the original video to output video with corrected audio

        Args:
            output_audio_filename (str, optional): Name of the file where the output audio will be saved. 
            Defaults to 'output_audio.wav'.

        Returns:
            tuple: Returns the original and updated transcript
        """  
              
        self.extract_audio()
        print('Transcribing audio')
        transcript, word_timestamps = self.transcribe_audio()
        print('Fixing mistakes')
        response = json.loads(self.get_response(transcript))
        response = response['choices'][0]['message']['content']

        print('Syncing audio')
        timed_transcript = self.align_transcripts(transcript.lower().split(),
                                                       word_timestamps, 
                                                       self._remove_punctuation(response).lower().split())
        
        concatenated_transcript = self.concatenate_simultaneous_words(timed_transcript)
        self.synthesize_audio(concatenated_transcript, f'Output/{output_audio_filename}', len(word_timestamps))

        print('Replacing audio')
        self.replace_audio(f'Output/{output_audio_filename}', 'Output/output.mp4')
        return transcript, response
