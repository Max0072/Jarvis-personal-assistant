from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from pynput import keyboard

load_dotenv()


class LongMemory:
    def __init__(self, client):
        self.client = client
        self.path = "memory.json"
        self.memory = json.load(open(self.path)) if os.path.exists(self.path) else []

    def save(self):
        json.dump(self.memory, open(self.path, "w"), ensure_ascii=False, indent=2)

    def embed(self, text):
        return self.client.embeddings.create(model="text-embedding-3-small", input=text).data[0].embedding

    def add(self, fact):
        self.memory.append({"text": fact, "embedding": self.embed(fact), "ts": time.time()})
        self.save()

    def search(self, query, top_k=3):
        if not self.memory:
            return []
        q = np.array(self.embed(query))
        scored = []
        for m in self.memory:
            scored.append((np.dot(q, m["embedding"]), m["text"]))
        scored.sort(reverse=True)
        return [text for _, text in scored[:top_k]]


class Chat:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.long_memory = LongMemory(self.client)
        self.history = []

    def build_messages(self, user_input):
        relevant = self.long_memory.search(user_input)
        system = "You are a helpful personal assistant."
        if relevant:
            system += "\n\nWhat you know about the user:\n" + "\n".join(f"- {f}" for f in relevant)
        return [{"role": "system", "content": system}] + self.history

    def extract_facts(self, user_input, reply):
        prompt = (
            f"Extract facts worth remembering long-term about the user (name, preferences, goals) "
            f"from this exchange. Return JSON: {{\"facts\": [...]}} or {{\"facts\": []}}.\n\n"
            f"User: {user_input}\nAssistant: {reply}"
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        for fact in data.get("facts", []):
            self.long_memory.add(fact)

    def send(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        res = self.client.chat.completions.create(model=self.model, messages=self.build_messages(user_input))
        reply = res.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        self.extract_facts(user_input, reply)
        return reply

    def listen(self, samplerate=16000):
        chunks = []

        def on_audio(indata, frames, t, status):
            chunks.append(indata.copy())

        print("Hold SPACE to speak, release to send...")
        with keyboard.Events() as events:
            for event in events:
                if isinstance(event, keyboard.Events.Press) and event.key == keyboard.Key.space:
                    break

        print("Recording...")
        with sd.InputStream(samplerate=samplerate, channels=1, dtype=np.int16, callback=on_audio):
            with keyboard.Events() as events:
                for event in events:
                    if isinstance(event, keyboard.Events.Release) and event.key == keyboard.Key.space:
                        break

        audio = np.concatenate(chunks)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, samplerate)
            transcription = self.client.audio.transcriptions.create(model="whisper-1", file=open(tmp_path, "rb"))
        finally:
            os.unlink(tmp_path)

        return transcription.text

    def speak(self, text):
        res = self.client.audio.speech.create(model="tts-1", voice="alloy", input=text, response_format="pcm")
        sd.play(np.frombuffer(res.content, dtype=np.int16), samplerate=24000)
        sd.wait()

    def run(self):
        print("Chat started. Press Ctrl+C to quit.\n")
        while True:
            user_input = self.listen()
            if not user_input.strip():
                continue
            print(f"You: {user_input}")
            reply = self.send(user_input)
            print(f"Agent: {reply}\n")
            self.speak(reply)


if __name__ == "__main__":
    Chat().run()
