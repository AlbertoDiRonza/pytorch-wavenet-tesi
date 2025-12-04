import json
import os
import shutil

toy_dir = "C:/Users/Alberto/Documents/GitHub/pytorch-wavenet-tesi/pytorch-wavenet-tesi/toy"
os.makedirs(toy_dir, exist_ok=True)

for filename in os.listdir("C:/Users/Alberto/Desktop/non committable/annotations"):
    if filename.startswith("sound-") and filename.endswith(".json"): 
        # Estrai la parte tra 'sound-' e '.json'
        file_id = filename[len("sound-"):-len(".json")]
        #entra nel file json
        #se contiene "house / techno"
        #allora prendi il file con quel id, ricostruisci 
        #id_numero.wav.wav e cerca in directory audio se viene trovato sposta in directory toy
        with open(os.path.join("C:/Users/Alberto/Desktop/non committable/annotations", f"sound-{file_id}.json")) as f:
            data = json.load(f)
            if ("house / techno" in data.get("genres", [])) \
                and (data.get("bpm") == "120"):
                #cerca il file audio
                for audio_filename in os.listdir("C:/Users/Alberto/Desktop/non committable/audio/wav"):
                    if audio_filename.startswith(file_id) and audio_filename.endswith(".wav"):
                        audio_path = os.path.join("C:/Users/Alberto/Desktop/non committable/audio/wav", f"{audio_filename}")
                        print(f"{audio_path}")
                        if os.path.exists(audio_path):
                            print(f"Copiato {audio_filename} in toy")
                            shutil.copy(audio_path, os.path.join(toy_dir, audio_filename))
                        else:
                            print(f"File audio {audio_filename} non trovato.")
            else:
                print("non trovato")
