from pydantic import BaseModel, Field
import os
import ollama
from typing import List, Optional
import json
import sounddevice as sd
import numpy as np
import queue
import sys
import requests
import re

import io
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel


try:
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    VOICE_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Voice recognition dependencies missing: {e}")
    print("To enable Whisper, run: pip install faster-whisper scipy sounddevice numpy")
    VOICE_AVAILABLE = False

OLLAMAADRESS = os.getenv("OLLAMAADDRESS", "http://192.168.250.220:11434")
OLLAMAMODEL = os.getenv("OLLAMAMODEL", "llama3:8b")

class OllamaChatbot:
    def __init__(self):
        try:
            self.client = ollama.Client(host=OLLAMAADRESS)
            print(f"[OllamaChatbot] New instance created successfully")
        except Exception as e:
            print(f"[OllamaChatbot] ERROR: Could not connect to {OLLAMAADRESS}. {e}")
            raise
        
    def reconfigure(self, context):
        self.context = context

    def sync_robot_state(self):
        """Fetch current board layout and update the system prompt context."""
        try:
            response = requests.get("http://localhost:8000/api/status")
            if response.status_code == 200:
                state = response.json()
                return state
        except Exception as e:
            print(f"[Ollama] Failed to sync state: {e}")

    def chat(self, newMessage:str, history:list):
        chatMessages = [
            {"role": "user", "content": newMessage}
        ]

        #for userMsg, chatBotMsg in history:
        #    chatMessages.append({"role": "user", "content": userMsg})
        #    chatMessages.append({"role": "assistant", "content": chatBotMsg})
        
        #chatMessages.append({"role": "user", "content": newMessage})
        chatData = [{'role': 'system', 'content': self.context}] + chatMessages
        
        try:
            response = self.client.chat(
                model=OLLAMAMODEL, 
                messages=chatData,
                format=ActionDescription.model_json_schema()
            )
            bot_content = response["message"]["content"]
            history.append([newMessage, bot_content])
            return bot_content, history
        except Exception as e:
            print(f"Error: {e}")
            return "", history
        
class Coordinate(BaseModel):
    m: int
    r: int
    c: int

class ActionItem(BaseModel):
    piece_id: str  # Kept as string to seamlessly accept "1", "2", "3", "4"
    from_coords: Coordinate = Field(alias="from")
    to_coords: Coordinate = Field(alias="to")

class ActionDescription(BaseModel):
    response: str = Field(description="A clear, friendly spoken response in English explaining what you are doing.")
    actions: List[ActionItem]
    
class FasterWhisperVoiceHandler:
    def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
        """
        Initializes the faster-whisper model.
        Available English models: 'tiny.en', 'base.en', 'small.en'
        device: 'cpu' or 'cuda' (if you have an NVIDIA GPU)
        """
        print(f"[Voice] Loading local Whisper model ({model_size}) on {device}...")
        # 'int8' optimization makes it run extremely fast on standard CPUs
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000
        print("[Voice] Local faster-whisper engine initialized successfully!")

    def _process_audio_buffer(self, recording_ndarray) -> str:
        """Helper to convert raw sounddevice array into a text transcription string."""
        # Convert audio array to standard WAV bytes in-memory
        wav_io = io.BytesIO()
        wavfile.write(wav_io, self.sample_rate, recording_ndarray)
        wav_io.seek(0)
        
        # Transcribe audio array stream
        segments, info = self.model.transcribe(wav_io, beam_size=5, language="en")
        
        # Merge structural audio segments back into a cohesive sentence string
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def listen_for_speech(self, duration=5) -> Optional[str]:
        """Captures a single static slice of voice command audio."""
        try:
            print(f"\n [Voice] LISTENING NOW ({duration}s)... Speak clearly!")
            # Record via sounddevice at the strict 16kHz Whisper requirement
            recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()
            
            print("[Voice] Analyzing speech patterns...")
            text = self._process_audio_buffer(recording)
            
            if text:
                print(f" [MIC CAPTURED] → \"{text}\"")
                return text
            else:
                print(" [MIC CAPTURED] → (nothing detected)")
                return None
        except Exception as e:
            print(f"[Voice] Transcription process failed: {e}")
            return None

    def listen_continuous(self, timeout=10) -> Optional[str]:
        """Maps seamlessly to your voice2 continuous tracking configuration loops."""
        return self.listen_for_speech(duration=timeout)

# --- NEW HOMOPHONE PRE-PROCESSOR ---
def clean_phonetics(raw_text: str) -> str:
    if not raw_text:
        return ""
    
    text = raw_text.lower().strip()

    text = re.sub(r'\b(peace)\b', 'piece', text)
    text = re.sub(r'\b(most|moose|more)\b', 'move', text)
    text = re.sub(r'\b(de|the)\b', 'D', text)
    text = re.sub(r'\b(be|beam)\b', 'B', text)
    
    # Catch structural homophones based on common STT error arrays
    text = re.sub(r'\b(won|wan|on|one)\b', '1', text)
    text = re.sub(r'\b(too|through|true|movies|two)\b', '2', text)
    text = re.sub(r'\b(tree|free|itree|three)\b', '3', text)
    text = re.sub(r'\b(for|fore|four)\b', '4', text)
    text = re.sub(r'\b(roll|road|rome|role|wrote|low|rail)\b', 'row', text)
    
    # Convert block/piece phrases straight to Letters for the LLM!
    text = text.replace("block 1", "piece A").replace("piece 1", "piece A")
    text = text.replace("block 2", "piece B").replace("piece 2", "piece B")
    text = text.replace("block 3", "piece C").replace("piece 3", "piece C")
    text = text.replace("block 4", "piece D").replace("piece 4", "piece D")
    
    
    return text

def execute_robot_move(piece_id, from_coords, to_coords):
    url = "http://localhost:8000/api/move_piece"
    payload = {"piece_id": piece_id, "from": from_coords, "to": to_coords, "speed": 0.3}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f" Move accepted! Job ID: {response.json().get('job_id')}")
            return True
        else:
            print(f" Server Error: {response.json().get('detail')}")
            return False
    except Exception as e:
        print(f" Connection failed: {e}")
        return False

def print_help():
    print("\n--- Commands Menu ---")
    print("  Type text OR use 'voice' / 'voice2' / 'mic' modes.")
    print("  Blocks are now tracked via numbers: 1, 2, 3, 4.")

if __name__ == "__main__":
    chatBot = OllamaChatbot()
    
    # CHANGED HERE: Re-engineered System Configuration context instructions targeting Numbers
    
    chatBot.reconfigure(
        "You are an intelligent UR3e robot controller node. You receive voice commands to move around pieces in a board with 2 matrixes, each with 2 columns and 4 rows."
        "Your job is to translate their voice intent into a clean, structured JSON payload matching this strict Pydantic schema structure:\n"

        "CRITICAL STATE DIRECTIONS:\n"
        "- NEVER assume or hardcode piece positions. ALWAYS extract the 'from' origin coordinates directly from the provided [CURRENT BOARD STATE] text mapping.\n"
        "- The pieces are uniquely identified as: 'A', 'B', 'C', or 'D'.\n"
        "- The board indices use standard human 1-indexed integers: m (matrix 1-2), r (row 1-4), c (column 1-2). Never use 0 or null.\n\n"

        "PARAMETER INHERITANCE RULES:\n"
        "- When the user requests a move but only specifies ONE or TWO target parameters (e.g., 'move piece A to matrix 2' or 'move piece A to row 2'), you must look up its current coordinate values in [CURRENT BOARD STATE] and keep the unspecified parameters EXACTLY the same.\n"
        "- Example: If A is at (m:1, r:2, c:2) and user says 'move to matrix 2', the 'to' coordinates MUST be (m:2, r:2, c:2). Do not pull old coordinates from earlier conversations.\n\n"
        
        "never move a piece to a cell already occupied by another one"

        "You give two outputs when you move pieces: " \
        "1.Feedback given to user, friendly and explaining your thought process given the speech input from the user"
        "2.2.Output JSON explicitly tracking this structural payload: "
        "{'actions': [{'piece_id': 'ID', 'from': {'m':M,'r':R,'c':C}, 'to': {'m':M,'r':R,'c':C}}]}"

        "COMPLEX COMMANDS WILL REQUIRE MULTI-STEP RESPONSES IN THESE CASES, DO THIS :"
        "-create a list of steps necessary and send them in the feedback to the user"
        "-each step will require user confirmation to be sure the physical robot is ready for the next step"
    )
    
    chat_history = []
    voice_enabled = False
    voice_handler = None
    
    if VOICE_AVAILABLE:
        try:
            try:
                default_input = sd.default.device[0] # Uses Linux system default mic
                if default_input is None or default_input < 0:
                    sd.default.device = None
            except Exception:
                pass
            
            voice_handler = FasterWhisperVoiceHandler(model_size="tiny.en", device="cpu")
            voice_enabled = True
        except Exception as e:
            print(f"Failed to initialize voice handler: {e}")
            voice_enabled = False
    
    mic_mode = False
    print("--- Unconstrained Chat Setup Engaged ---")
    
    while True:
        current_status = chatBot.sync_robot_state() 
        skip_ai = False


        if mic_mode and voice_enabled:
            user_input = voice_handler.listen_continuous(timeout=5)
            if user_input:
                print(f"\nYou (voice-raw): {user_input}")
            else:
                continue
        else:
            user_input = input("\nYou (text): ").strip()

        cmd = user_input.lower()
        if cmd in ["quit", "exit", "sair"]:
            break
        elif cmd == "help":
            print_help()
            skip_ai = True
        elif cmd == "mic" and voice_enabled:
            mic_mode = not mic_mode
            print(f"\n[Continuous Mic Mode: {'ON' if mic_mode else 'OFF'}]")
            skip_ai = True
        elif cmd == "voice" and voice_enabled:
            user_input = voice_handler.listen_for_speech(duration=5)
            if user_input:
                print(f"\nYou (voice-raw): {user_input}")
            skip_ai = False if user_input else True
        elif cmd == "voice2" and voice_enabled:
            user_input = voice_handler.listen_continuous(timeout=10)
            if user_input:
                print(f"\nYou (voice-raw): {user_input}")
            skip_ai = False if user_input else True
        elif cmd == "board":
            if current_status and "present_board" in current_status:
                board = current_status["present_board"]
                print("\n" + "="*45)
                print(" CURRENT BOARD POSITIONS")
                print("="*45)
                # Sort alphabetically by piece letter ID for clean readability
                for p in sorted(board, key=lambda x: x['id']):
                    print(f"  • Piece {p['id']}  ➔  Matrix: {p['m']} | Row: {p['r']} | Column: {p['c']}")
                print("="*45)
            else:
                print("\n Error: Unable to fetch live status. Is server.py online?")
            
            skip_ai = True  # Block Ollama from processing this command turn
        
        # --- PROCESSING INTERCEPT PHASE ---
        if user_input and not skip_ai:
            cleaned_input = clean_phonetics(user_input)
            
            board_desc = "Unknown"
            if current_status and "present_board" in current_status:
                board = current_status["present_board"]
                board_desc = ", ".join([f"Piece {p['id']} at (Matrix {p['m']}, Row {p['r']}, Column {p['c']})" for p in board])
            
            ai_payload = f"[CURRENT BOARD STATE]: {board_desc}\n[USER COMMAND]: {cleaned_input}"
            
            bot_response, chat_history = chatBot.chat(ai_payload, chat_history)
            print(f"\nYou (voice-Cleaned): {cleaned_input}")
            
            if bot_response:
                try:
                    action_obj = ActionDescription.model_validate_json(bot_response)
                    print(f"\n Robot says: \"{action_obj.response}\"")
                    print("\n--- Sending JSON Actions to Python Server ---")
                    
                    move_success = True
                    for move in action_obj.actions:
                        if not move.piece_id or move.piece_id.strip() == "":
                            continue
                            
                        f_coords = move.from_coords.model_dump()
                        t_coords = move.to_coords.model_dump()
                        
                        # Execute and track status
                        status_ok = execute_robot_move(move.piece_id, f_coords, t_coords)
                        if not status_ok:
                            move_success = False
                    
                    # CRITICAL FIX: If a move failed due to Desync, wipe it from chat history 
                    # so the LLM doesn't learn from its bad coordinates next turn!
                    if not move_success and len(chat_history) > 0:
                        print("[Ollama] Wiping failed turn from memory to prevent coordinate drift...")
                        chat_history.pop() 

                except Exception as e:
                    print(f"Response mapping skipped or validation mismatch: {e}")
                    if len(chat_history) > 0:
                        chat_history.pop()