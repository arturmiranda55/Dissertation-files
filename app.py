import streamlit as st
import ollama
import os
import io
import re
import asyncio
import json
import subprocess
import sys
import time

# Import MCP client capabilities natively into the UI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- AUDIO & SPEECH DEPENDENCIES ---
try:
    import sounddevice as sd
    import numpy as np
    from scipy.io import wavfile
    from faster_whisper import WhisperModel
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# --- CONFIGURATION ---
OLLAMAADRESS = os.getenv("OLLAMAADDRESS", "http://192.168.250.220:11434")
OLLAMAMODEL = os.getenv("OLLAMAMODEL", "llama3.1:8b")

SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-u", "robot_mcp_server.py"],
    env={**os.environ}
)

# --- INITIALIZE PAGE CONFIG FIRST ---
if 'setup_done' not in st.session_state:
    st.set_page_config(page_title="InterBot Orchestrator", layout="wide")
    st.session_state.setup_done = True



# --- CUSTOM CSS FOR STYLIZED UI ---
st.markdown("""
    <style>
    /* 1. Sidebar Width set to 600px */
    [data-testid="stSidebar"] {
        min-width: 500px !important;
        max-width: 500px !important;
        background-color: #F8FAFC;
    }
    
    /* Logo Image Centering & Spacing */
    [data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        padding-bottom: 10px !important;
    }
    
    /* Chunky Red "Create Agent" Button */
    div.stButton > button:first-child {
        background-color: #E11D48 !important;
        color: white !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        padding: 16px 24px !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(225, 29, 72, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #BE123C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(225, 29, 72, 0.45) !important;
    }
    
    /* 2. Transform Sidebar Radio Items into Chunky Agent Cards */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        width: 100% !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
    }
    
    /* Hover effect for agent cards */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        border-color: #94A3B8 !important;
        transform: translateX(4px) !important;
    }
    
    /* Selected agent card highlight */
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #EFF6FF !important;
        border-color: #2563EB !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* Enlarge the text inside the agent cards */
    [data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #1E293B !important;
    }
    </style>
""", unsafe_allow_html=True)



# --- INITIALIZE VOICE RECOGNITION ONCE ---
if VOICE_AVAILABLE and "voice_handler" not in st.session_state:
    class StreamlitWhisperHandler:
        def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.sample_rate = 16000
        def _process_audio_buffer(self, recording_ndarray):
            wav_io = io.BytesIO()
            wavfile.write(wav_io, self.sample_rate, recording_ndarray)
            wav_io.seek(0)
            segments, _ = self.model.transcribe(wav_io, beam_size=5, language="en")
            return " ".join([segment.text for segment in segments]).strip()
        def listen_for_speech(self, duration=5):
            recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype='int16')
            sd.wait()
            return self._process_audio_buffer(recording)
            
    st.session_state.voice_handler = StreamlitWhisperHandler()

MAX_RETRIES = 3

def clean_phonetics(raw_text: str) -> str:
    if not raw_text: return ""
    text = raw_text.lower().strip()
    text = re.sub(r'\b(peace)\b', 'piece', text)
    text = re.sub(r'\b(most|moose|more)\b', 'move', text)
    text = re.sub(r'\b(won|wan|on|one)\b', '1', text)
    text = re.sub(r'\b(too|through|true|two)\b', '2', text)
    text = re.sub(r'\b(tree|free|three)\b', '3', text)
    return text

# --- LIVE HARDWARE MCP PIPELINE OPERATORS ---
async def fetch_board_resource_live() -> str:
    try:
        async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                board_resource = await session.read_resource("robot://board/state")
                return board_resource.contents[0].text
    except Exception as e:
        return f"Offline: Live grid unavailable ({e})"

async def execute_mcp_tool_live(tool_name: str, tool_args: dict) -> str:
    try:
        async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=tool_args)
                return result.content[0].text
    except Exception as e:
        return f"Hardware Execution Failure: {e}"

# --- INITIALIZE MULTI-AGENT META-FACTORY TEAM ---
if "agents" not in st.session_state:
    st.session_state.agents = {
        "user_agent": {
            "purpose": "Analyze provided backend codebases dynamically to construct hardware operational manifests.",
            "prompt": (
                "You are the Autonomous Systems Analysis Agent. Your job is to audit unknown directives for other agents "
                "provided by the user to build a short integration blueprint.\n\n"
                
                "YOUR ANALYSIS WORKFLOW:\n"
                "1. Get the new agents name.\n"
                "2. Collect the new agents purpose and the context for the workplace .\n"
                "3. Instruct that the agent must always use his tool, if possible"
                
                "DELIVERY MANDATE:\n"
                "- YOU MUST INCLUDE THE NAME TAG IN YOUR FINAL OUTPUT: Write `[config: name=agent_name_here]` replacing `agent_name_here` with a clean lowercase string (alphanumeric and underscores only).\n"
                "- Conclude your message text with the exact phrase 'DEPLOY_TEAM' to trigger the compilation loop."
            ),
            "history": [],
            "has_tools": False
        },
        "orchestrator": {
            "purpose": "Manages workspace orchestration and synthesizes custom worker identities.",
            "prompt": (
                "You are the Meta-Orchestrator. You receive a hardware manifest and human objective, "
                "and your job is to synthesize specialized, context-aware worker identities.\n\n"
                "YOUR TASKS:\n"
                "1. Direct the Code Agent to construct a FastMCP server mapping the provided API schema.\n"
                "2. Synthesize a specialized SYSTEM PROMPT for the runtime Worker Agent derived ENTIRELY from "
                "the provided hardware interface summary and human goal.\n\n"
                "UNIVERSAL WORKER PROMPT DESIGN RULES:\n"
                "The synthesized system prompt MUST strictly instruct the generated worker agent to:\n"
                "- NATIVE TOOL CALLING: Invoke functions natively through the system's tool interface. "
                "- NEVER output markdown code blocks, JSON strings, or pseudo-code simulating tool execution (e.g. '{\"name\": ...}').\n"
                "- CONTEXT UTILIZATION: Extract real-time environmental context provided in message headers (e.g. state summaries or resource logs) "
                "to resolve function arguments before calling action tools.\n"
                "- STRICT TYPE INTEGRITY: Adhere strictly to the parameter types in the tool schemas:"
                "- ABSOLUTELY NEVER send null or NoneTypes or NoveValues for required fields, always use valid values.\n"
                "- DIRECT ACTION: Execute tool calls immediately once user intent and target arguments are clear, without asking unnecessary conversational confirmation.\n\n"
                "OUTPUT FORMAT MANDATE:\n"
                "You must wrap the synthesized worker system prompt strictly between structural tags:\n"
                "[WORKER_PROMPT_START]\n"
                "You are [Agent Name], a specialized assistant for [Objective]...\n"
                "[WORKER_PROMPT_END]"
            ),
            "history": [],
            "has_tools": True
        },
        "code_agent": {
            "purpose": "Translates hardware API schemas into non-blocking FastMCP stdio bridge servers dynamically.",
            "prompt": (
                "You are the FastMCP Code Compiler. Translate incoming API schemas into a standalone Python FastMCP server script.\n\n"
                "CRITICAL ARCHITECTURE RULES:\n"
                "1. NO CLASSES OR SUBCLASSING: NEVER write 'class ...', NEVER subclass FastMCP, and NEVER use 'self'. Write ONLY top-level standalone functions.\n"
                "2. INSTANTIATION: Instantiate FastMCP at the top-level module scope first:\n"
                "   `mcp = FastMCP('Dynamic_Hardware_Bridge')`\n"
                "3. DECORATORS: Use ONLY `@mcp.tool()` on top-level `async def` functions.\n"
                "4. IMPORTS: Import ONLY `from mcp.server.fastmcp import FastMCP`, `import requests`, `import json`, `import os`, `import anyio`.\n"
                "5. SIGNATURES: Use flat primitive types (`int`, `str`, `float`) for function parameters. Do not pass Pydantic models or dicts.\n"
                "6. ENTRYPOINT: End the script with:\n"
                "   if __name__ == '__main__':\n"
                "       mcp.run()\n\n"
                "OUTPUT FORMAT: Respond strictly with a single Python code block (```python ... ```)."
            ),
            "history": [
                {
                    "role": "user",
                    "content": (
                        "Synthesize a FastMCP server mapping the provided API schema endpoints.\n\n"
                        "SCHEMA CONTEXT:\n"
                        "{\n"
                        "  \"base_url\": \"http://localhost:8000\",\n"
                        "  \"endpoints\": [\n"
                        "    {\"path\": \"/api/status\", \"method\": \"GET\", \"type\": \"resource\", \"uri\": \"system://board/status\"},\n"
                        "    {\"path\": \"/api/execute_command\", \"method\": \"POST\", \"type\": \"tool\", \"params\": {\"command\": \"str\", \"intensity\": \"float\"}}\n"
                        "  ]\n"
                        "}"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "```python\n"
                        "from mcp.server.fastmcp import FastMCP\n"
                        "import requests\n"
                        "import json\n"
                        "import os\n"
                        "import anyio\n\n"
                        "BASE_URL = os.getenv(\"TARGET_API_URL\", \"http://localhost:8000\")\n"
                        "mcp = FastMCP(\"Dynamic_Hardware_Bridge\")\n\n"
                        "@mcp.resource(\"system://board/status\")\n"
                        "async def get_system_status() -> str:\n"
                        "    \"\"\"Fetches system status from backend endpoint.\"\"\"\n"
                        "    url = f\"{BASE_URL}/api/status\"\n"
                        "    try:\n"
                        "        response = await anyio.to_thread.run_sync(lambda: requests.get(url, timeout=3.0))\n"
                        "        if response.status_code == 200:\n"
                        "            return response.text\n"
                        "        return json.dumps({\"error\": f\"API HTTP {response.status_code}\"})\n"
                        "    except Exception as e:\n"
                        "        return json.dumps({\"error\": f\"Endpoint unreachable: {str(e)}\"})\n\n"
                        "@mcp.tool()\n"
                        "async def execute_command(command: str, intensity: float) -> str:\n"
                        "    \"\"\"Dispatches command payload to the target device.\"\"\"\n"
                        "    payload = {\"command\": command, \"intensity\": intensity}\n"
                        "    url = f\"{BASE_URL}/api/execute_command\"\n"
                        "    try:\n"
                        "        response = await anyio.to_thread.run_sync(lambda: requests.post(url, json=payload, timeout=10.0))\n"
                        "        if response.status_code == 200:\n"
                        "            return f\"Success: Executed {command}.\"\n"
                        "        return f\"Rejection: {response.text}\"\n"
                        "    except Exception as e:\n"
                        "        return f\"Network Error: {str(e)}\"\n\n"
                        "if __name__ == \"__main__\":\n"
                        "    mcp.run()\n"
                        "```"
                    )
                }
            ],
            "has_tools": False
        },
        "reviewer_agent": {
            "purpose": "Audits generated FastMCP Python files for protocol accuracy.",
            "prompt": (
                "You are the Core Code Reviewer. Audit the Code Agent's generated Python FastMCP code against these factory standards:\n"
                "1. Does it correctly import `FastMCP` and instantiate it?\n"
                "2. Does it accurately handle the outbound JSON structure requested by the chief?\n"
                "3. Does it gracefully catch connection timeouts/exceptions when contacting the physical robot API?\n"
                "4. Does it cleanly return both the updated board state string and a confirmation message?\n\n"
                "Respond with 'APPROVED' at the top if it meets all standards. Otherwise, provide a structured list of compilation errors for the Code Agent to fix."
                "Does the @mcp.tool() function accept raw primitive parameters (strings/ints)? If it accepts a Pydantic object or dictionary directly as a function argument, REJECT IT IMMEDIATELY."
            ),
            "history": [],
            "has_tools": False
        }
    }

if "active_agent" not in st.session_state:
    st.session_state.active_agent = "user_agent"

async def fetch_mcp_tools_dynamic():
    try:
        async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                manifest = await session.list_tools()
                
                ollama_formatted_tools = []
                for tool in manifest.tools:
                    ollama_formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                return ollama_formatted_tools
    except Exception as e:
        return None

def extract_aas_interface_summary(raw_file_str: str) -> str:
    if not raw_file_str: return "No hardware schema attached."
    try:
        data = json.loads(raw_file_str)
        if isinstance(data, dict) and "endpoints" in data:
            return json.dumps([{"path": ep.get("path"), "method": ep.get("method"), "params": ep.get("params")} for ep in data["endpoints"]], indent=2)
    except Exception:
        pass
    return raw_file_str[:800]

def run_agent_inference(active_name, system_prompt, chat_history, user_message, pass_tools=False):
    client = ollama.Client(host=OLLAMAADRESS)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})
    
    ollama_tools = None
    if pass_tools and active_name not in ["user_agent", "orchestrator", "code_agent", "reviewer_agent"]:
        ollama_tools = asyncio.run(fetch_mcp_tools_dynamic())

    try:
        response = client.chat(model=OLLAMAMODEL, messages=messages, tools=ollama_tools)
        return response["message"]
    except Exception as e:
        return {"role": "assistant", "content": f"Ollama Core Execution Error: {e}"}

# --- STYLIZED SIDEBAR REGISTRY ---
with st.sidebar:
    # 1. Logo Display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("InterBot.png", width="stretch")
    
    # 2. Chunky Red "Create Agent" Button
    if st.button("➕ Create Agent", width="stretch"):
        st.session_state.active_agent = "user_agent"
        if "sidebar_agent_radio" in st.session_state:
            del st.session_state["sidebar_agent_radio"]
        st.rerun()

    st.markdown("---")
    st.subheader("Active Agents")
    
    all_keys = list(st.session_state.agents.keys())
    
    # Strictly exclude user_agent AND internal backend agents from the radio list
    ui_visible_options = [
        k for k in all_keys 
        if k not in ["user_agent", "orchestrator", "code_agent", "reviewer_agent"]
    ]
    
    if ui_visible_options:
        # Determine initial selection index safely
        if st.session_state.active_agent in ui_visible_options:
            current_idx = ui_visible_options.index(st.session_state.active_agent)
        else:
            current_idx = 0

        # Dedicated callback updates state cleanly upon user click
        def on_agent_select():
            st.session_state.active_agent = st.session_state.sidebar_agent_radio

        # Render radio only if user_agent is NOT the active workspace view
        if st.session_state.active_agent != "user_agent":
            st.radio(
                "Select Deployed Worker:",
                options=ui_visible_options,
                index=current_idx,
                format_func=lambda x: f"🤖 {x.upper().replace('_', ' ')}",
                key="sidebar_agent_radio",
                on_change=on_agent_select
            )
        else:
            # Show options unselected when in creation mode
            st.caption("Select a worker below to view its chat:")
            for op in ui_visible_options:
                if st.button(f"🤖 {op.upper().replace('_', ' ')}", key=f"btn_{op}", use_container_width=True):
                    st.session_state.active_agent = op
                    st.rerun()

    else:
        st.caption("No deployed agents yet. Click 'Create Agent' above to get started!")



# --- LAYOUT DESIGN: CENTER DIALOGUE AREA ---
active_name = st.session_state.active_agent
active_agent = st.session_state.agents[active_name]

if active_name == "user_agent":
    st.title("➕ Create New Agent")
    st.caption("Analyze backend codebases or hardware schemas to construct a new dedicated Worker Agent.")
else:
    st.title(f"🤖 {active_name.upper().replace('_', ' ')}")
    st.info(f"{active_agent['purpose']}")

# --- FILE UPLOADER FOR CREATION AGENT ---
raw_file = ""
if active_name == "user_agent":
    uploaded_file = st.file_uploader(
        "Drag and drop your server.py, api.py, or hardware schema configuration here", 
        type=["py", "json", "txt", "md"]
    )
    if uploaded_file is not None:
        st.session_state["raw_file_contents"] = uploaded_file.read().decode("utf-8")
        st.success(f"Successfully annexed {uploaded_file.name}!")

# Render historical chats
for msg in active_agent["history"]:
    if msg.get("content"):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# INPUT HANDLING
processed_input = None

if active_agent.get("has_tools"):
    bound_tools = asyncio.run(fetch_mcp_tools_dynamic())
    if bound_tools:
        tool_names = [t["function"]["name"] for t in bound_tools]
        st.success(f" Connected to MCP Server: **{len(bound_tools)} Tool(s) Loaded** (`{', '.join(tool_names)}`)")

if user_query := st.chat_input(f"Transmit instruction to {active_name}..."):
    processed_input = user_query

if VOICE_AVAILABLE:
    col1, _ = st.columns([1, 4])
    with col1:
        if st.button("🎤 Record Voice (5s)", use_container_width=True):
            with st.spinner(" Listening..."):
                raw_voice = st.session_state.voice_handler.listen_for_speech(duration=5)
            if raw_voice:
                processed_input = clean_phonetics(raw_voice)
                st.success(f"Transcribed Input: \"{processed_input}\"")

if active_name == "user_agent" and st.session_state.get("raw_file_contents") and st.button("🚀 Process Annexed File Now"):
    processed_input = "Analyze the attached codebase file."

# --- LIVE INFERENCE ROUTING LOGIC ---
if processed_input:
    payload_prompt = processed_input


    # Render User Query Bubble
    with st.chat_message("user"):
        st.write(processed_input)
    active_agent["history"].append({"role": "user", "content": processed_input})

    # Generate Response Panel
    with st.chat_message("assistant"):
        
        if active_name == "user_agent":

            # Retrieve raw uploaded file context if available
            raw_context = st.session_state.get("raw_file_contents", "")
            
            # Construct payload including the file context
            if raw_context and raw_context not in payload_prompt:
                full_user_payload = (
                    f"ATTACHED FILE / SCHEMA CONTEXT:\n```\n{raw_context}\n```\n\n"
                    f"USER DIRECTIVE: {payload_prompt}"
                )
            else:
                full_user_payload = payload_prompt


            with st.spinner("User Agent analyzing specifications..."):
                bot_msg = run_agent_inference(
                    "user_agent",
                    st.session_state.agents["user_agent"]["prompt"],
                    st.session_state.agents["user_agent"]["history"][:-1],
                    payload_prompt
                )
            
            user_agent_text = bot_msg.get("content", "")
            
            if "DEPLOY_TEAM" in user_agent_text:
                st.write(user_agent_text.replace("DEPLOY_TEAM", ""))
                st.info("⚡ Specification Checklist Met! Activating engineering team...")

                raw_file = st.session_state.get("raw_file_contents", "")
                
                with st.status(" Collaborative Squad Thinking...", expanded=True) as status:
                    st.write(" Orchestrator synthesizing worker identity prompt...")

                    orchestrator_payload = (
                        f"User Objective Summary: {user_agent_text}\n\n"
                        f"HARDWARE INTERFACE SPECIFICATIONS:\n{raw_file}"
                    )
                    
                    orchestrator_reply = run_agent_inference(
                        "orchestrator", 
                        st.session_state.agents["orchestrator"]["prompt"], 
                        [], 
                        orchestrator_payload
                    )
                    orchestrator_text = orchestrator_reply.get("content", "")

                    st.write(" Code Agent compiling FastMCP Server using direct AAS context...")

                    code_generation_prompt = (
                        f"Orchestrator Directive: {orchestrator_text}\n\n"
                        f"DIRECT TARGET HARDWARE REFERENCE CONTEXT:\n```json\n{raw_file}\n```\n"
                        f"Generate the exact python code matching this hardware schema specification. "
                        f"Ensure you use @mcp.tool() decorators with flat primitives as parameters."
                    )
                    
                    code_reply = run_agent_inference(
                        "code_agent", 
                        st.session_state.agents["code_agent"]["prompt"], 
                        st.session_state.agents["code_agent"]["history"], 
                        code_generation_prompt
                    )

                    code_text = code_reply.get("content", "")

                    # Extracting clean code block for the test file execution path
                    code_blocks = re.findall(r"```python\s*(.*?)\s*```", code_text, re.DOTALL)
                    extracted_python = code_blocks[0].strip() if code_blocks else code_text.strip()                    
                    
                    test_filename = "test_mcp_server.py"
                    with open(test_filename, "w", encoding="utf-8") as f:
                        f.write(extracted_python)                    

                    # Run background dry-run runtime verification safely
                    test_passed = True
                    error_message = ""
                    try:
                        st.write("Testing generated MCP server in background sandbox...")
                        process = subprocess.Popen(
                            [sys.executable, "-u", test_filename],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        time.sleep(2.5)
                        return_code = process.poll()
                        if return_code is not None and return_code != 0:
                            _, stderr_logs = process.communicate()
                            raise Exception(stderr_logs)
                        else:
                            process.terminate()
                            st.write("✨ Code passed sandbox execution check!")
                    except Exception as e:
                        test_passed = False
                        error_message = str(e)

                    if not test_passed:
                        status.update(label=" Sandbox Execution Test Failed!", state="error")
                        st.error(f"The Code Agent generated syntax or runtime errors!\n\n**Terminal Output:**\n```text\n{error_message}\n```")
                        st.stop()

                    st.write(" Reviewer Agent running protocol and syntax safety check...")
                    reviewer_reply = run_agent_inference(
                        "reviewer_agent", 
                        st.session_state.agents["reviewer_agent"]["prompt"], 
                        st.session_state.agents["reviewer_agent"]["history"], 
                        f"Code Proposal:\n{code_text}\n\nVerify against reference asset specification:\n{raw_file}"
                    )
                    reviewer_text = reviewer_reply.get("content", "")

                    if "APPROVED" in reviewer_text.upper():
                        status.update(label=" Code verified and approved by team!", state="complete")
                        
                        # 1. Primary: Search for strict tag [config: name=...]
                        name_match = re.search(r"\[config:\s*name=([a-zA-Z0-9_]+)\]", user_agent_text, re.IGNORECASE)
                        
                        if name_match:
                            agent_name = name_match.group(1).lower().strip()
                        else:
                            # 2. Fallback: Flexible regex for phrases like "called X", "named X", or "name: X"
                            text_search = re.search(r"(?:called|named|name[:=])\s*['\"]?([a-zA-Z0-9_]+)['\"]?", user_agent_text, re.IGNORECASE)
                            if text_search:
                                agent_name = text_search.group(1).lower().strip()
                            else:
                                # 3. Fallback: Generate unique incremental default name
                                existing_agents = [k for k in st.session_state.agents.keys() if k.startswith("agent")]
                                agent_name = f"agent{len(existing_agents) + 1}"
                        
                        # Dynamically extract the Orchestrator's custom-built prompt
                        prompt_match = re.search(r"\[WORKER_PROMPT_START\](.*?)\[WORKER_PROMPT_END\]", orchestrator_text, re.DOTALL)
                        if prompt_match:
                            dynamic_worker_prompt = prompt_match.group(1).strip()
                        else:
                            dynamic_worker_prompt = (
                                f"You are {agent_name.upper()}, a robotic assistant. "
                                f"Use your FastMCP tools to fulfill the following context: {user_agent_text}"
                            )
                        
                        # Overwrite the runtime production file channel
                        with open("robot_mcp_server.py", "w", encoding="utf-8") as f:
                            f.write(extracted_python)
                        
                        # Instantiate workspace memory tab dynamically
                        st.session_state.agents[agent_name] = {
                            "purpose": f"Dynamic agent customized for user task profile.",
                            "prompt": dynamic_worker_prompt,
                            "history": [{"role": "assistant", "content": f"Hello! I am {agent_name}, configured specifically for your current application profile. Ready to assist!"}],
                            "has_tools": True 
                        }
                        
                        final_summary = f" 🎉 **Deployment Successful!**\n\n- New Node: `{agent_name}`\n- Identity Prompt Status: **Dynamically Generated by Orchestrator**\n\nSwitch tabs in the sidebar to see your custom agent."
                        st.success(final_summary)
                        st.session_state.agents["user_agent"]["history"].append({"role": "assistant", "content": final_summary})
                    else:
                        status.update(label=" Reviewer rejected the proposal safety check.", state="error")
                        st.error(f"Reviewer Feedback:\n{reviewer_text}")
                        st.session_state.agents["user_agent"]["history"].append({"role": "assistant", "content": f"Deployment halted due to review errors:\n{reviewer_text}"})
                        
                        st.session_state.active_agent = agent_name
            else:
                if user_agent_text:
                    st.write(user_agent_text)
                    st.session_state.agents["user_agent"]["history"].append(bot_msg)
                else:
                    st.error("Error: Received empty response from model.")

        else:
            with st.spinner("Fetching live board resource state..."):
                live_board = asyncio.run(fetch_board_resource_live())
                worker_payload = f"[CURRENT PHYSICAL BOARD RESOURCE STATE]: {live_board}\n\n[USER INSTRUCTION]: {processed_input}"

            with st.spinner("Inference in progress..."):
                bot_msg = run_agent_inference(
                    active_name, 
                    active_agent["prompt"], 
                    active_agent["history"][:-1], 
                    worker_payload, 
                    pass_tools=active_agent["has_tools"]
                )
                
                if "tool_calls" in bot_msg:
                    for tool_call in bot_msg["tool_calls"]:
                        t_name = tool_call["function"]["name"]
                        t_args = tool_call["function"]["arguments"]
                        
                        st.warning(f" [MCP Execution] Dispatching payload data parameters: {t_args}")
                        hw_response = asyncio.run(execute_mcp_tool_live(t_name, t_args))
                        st.success(f" Callback Result: {hw_response}")
                        active_agent["history"].append({"role": "assistant", "content": f"[Executed {t_name}]: {hw_response}"})
                else:
                    st.write(bot_msg["content"])
                    active_agent["history"].append(bot_msg)
                
    st.rerun()