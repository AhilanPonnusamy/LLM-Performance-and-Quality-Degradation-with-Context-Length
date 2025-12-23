import os
import csv
import time
import threading
import psutil
import random
from nvitop import Device
from openai import OpenAI
from datetime import datetime

# --- CONFIGURATION ---
# Model name must exactly match the model loaded in the vLLM server
MODEL_BEING_TESTED = "mistralai/Mixtral-8x7B-Instruct-v0.1"
VLLM_HOST = "http://vllm:8000/v1"
API_KEY = "vllm"  # Dummy key for vLLM

STARTING_THREADS = 1 # set the Initial Thread count
THREAD_INCREMENT = 1
MAX_THREADS_SAFETY_LIMIT = 100  # A safety limit to prevent an infinite loop
GPU_UTILIZATION_THRESHOLD = 99.95  # The target GPU % to stop the test

COOLING_PERIOD_S = 30
OUTPUT_CSV_FILE = "/app/results/results_vllm_mistral_4096.csv"  # Save to mounted volume

# Test configuration
HAYSTACK_FILE = "/app/haystack_text.txt"
SIMPLE_BASELINE_LOOPS = 10
NEEDLE_HAYSTACK_LOOPS = 10
# CONTEXT_TOKEN_PERCENTAGE = 0.50
MODEL_MAX_CONTEXT = 32768
MAX_WORDS_REQUIRED=4096 #0,4096,10000,15000
GENERAL_KNOWLEDGE_QUESTIONS = [
{"question": "What is the capital of France?", "answer": "paris"},
{"question": "Who wrote 'Hamlet'?", "answer": "shakespeare"},
{"question": "What is the chemical symbol for water?", "answer": "h2o"},
{"question": "In which year did the Titanic sink?", "answer": "1912"},
{"question": "What planet is known as the Red Planet?", "answer": "mars"},
{"question": "Who painted the Mona Lisa?", "answer": "vinci"},
{"question": "What is the tallest mountain in the world?", "answer": "everest"},
{"question": "What is the main ingredient in guacamole?", "answer": "avocado"},ssh-keygen -f "/home/laptop-obs-150/.ssh/known_hosts" -R "50.17.22.207"
{"question": "How many continents are there?", "answer": "seven"},
{"question": "Who was the first person to walk on the moon?", "answer": "armstrong"},
{"question": "What is the currency of Japan?", "answer": "yen"},
{"question": "What is the hardest natural substance on Earth?", "answer": "diamond"},
{"question": "Which ocean is the largest?", "answer": "pacific"},
{"question": "Who invented the telephone?", "answer": "bell"},
{"question": "What is the square root of 64?", "answer": "8"},
{"question": "Which country is famous for its pyramids?", "answer": "egypt"},
{"question": "What is the primary language spoken in Brazil?", "answer": "portuguese"},
{"question": "Who discovered penicillin?", "answer": "fleming"},
{"question": "What is the boiling point of water at sea level?", "answer": "100"},
{"question": "Which artist cut off his own ear?", "answer": "van gogh"},
{"question": "What is the largest animal in the world?", "answer": "blue whale"},
{"question": "In what country would you find the Eiffel Tower?", "answer": "france"},
{"question": "What is the name of the galaxy we live in?", "answer": "milky way"},
{"question": "How many sides does a triangle have?", "answer": "three"},
{"question": "Who is the author of the Harry Potter series?", "answer": "rowling"},
{"question": "What is the chemical symbol for gold?", "answer": "au"},
{"question": "What is the capital of Italy?", "answer": "rome"},
{"question": "Who was the first President of the United States?", "answer": "washington"},
{"question": "What gas do plants absorb from the atmosphere?", "answer": "carbon dioxide"},
{"question": "In which decade was the Internet invented?", "answer": "1960s"},
{"question": "What is the largest planet in our solar system?", "answer": "jupiter"},
{"question": "Who wrote 'To Kill a Mockingbird'?", "answer": "lee"},
{"question": "What is the currency of China?", "answer": "yuan"},
{"question": "What is the process by which plants make their food?", "answer": "photosynthesis"},
{"question": "Which animal is the largest primate?", "answer": "gorilla"},
{"question": "What is the capital of Spain?", "answer": "madrid"},
{"question": "What color is a ruby?", "answer": "red"},
{"question": "What is the longest river in the world?", "answer": "nile"},
{"question": "Who developed the theory of relativity?", "answer": "einstein"},
{"question": "How many states are there in the United States?", "answer": "50"},
{"question": "What is the highest-grossing film of all time (original release)?", "answer": "avatar"},
{"question": "What is the chemical symbol for oxygen?", "answer": "o"},
{"question": "Which element has the atomic number 1?", "answer": "hydrogen"},
{"question": "What is the name of the main villain in the 'Lord of the Rings' trilogy?", "answer": "sauron"},
{"question": "What is the capital of Germany?", "answer": "berlin"},
{"question": "What is the largest organ in the human body?", "answer": "skin"},
{"question": "Who painted the ceiling of the Sistine Chapel?", "answer": "michelangelo"},
{"question": "What is the primary gas in Earth's atmosphere?", "answer": "nitrogen"},
{"question": "What year did World War I begin?", "answer": "1914"},
{"question": "Which instrument is used to measure earthquakes?", "answer": "seismograph"},
{"question": "What is the name of the fictional detective who lives at 221B Baker Street?", "answer": "holmes"},
{"question": "What is the capital of Russia?", "answer": "moscow"},
{"question": "What is the chemical symbol for salt?", "answer": "nacl"},
{"question": "Who was the first woman to win a Nobel Prize?", "answer": "curie"},
{"question": "What is the largest desert in the world?", "answer": "sahara"},
{"question": "What metal is the best conductor of electricity?", "answer": "silver"},
{"question": "What is the name of the famous clock tower in London?", "answer": "big ben"},
{"question": "What is the capital of Canada?", "answer": "ottawa"},
{"question": "Which planet is closest to the sun?", "answer": "mercury"},
{"question": "Who wrote '1984'?", "answer": "orwell"},
{"question": "What is the unit of electric current?", "answer": "ampere"},
{"question": "What is the world's most populous country?", "answer": "china"},
{"question": "What is the chemical symbol for iron?", "answer": "fe"},
{"question": "Who was the Greek god of the sea?", "answer": "poseidon"},
{"question": "How many bones are in the adult human body?", "answer": "206"},
{"question": "What city hosted the 2000 Summer Olympics?", "answer": "sydney"},
{"question": "What famous German composer wrote the 'Ninth Symphony'?", "answer": "beethoven"},
{"question": "What is the main chemical compound in natural gas?", "answer": "methane"},
{"question": "Which war ended in 1945?", "answer": "world war ii"},
{"question": "What is the main character's name in 'The Great Gatsby'?", "answer": "gatsby"},
{"question": "What is the capital of Australia?", "answer": "canberra"},
{"question": "What is the boiling point of the Celsius scale?", "answer": "100"},
{"question": "Who invented the light bulb?", "answer": "edison"},
{"question": "What type of star is the sun?", "answer": "g-type main-sequence"},
{"question": "What is the formula for calculating force?", "answer": "f=ma"},
{"question": "What is the name of the first satellite launched into space?", "answer": "sputnik"},
{"question": "What is the smallest country in the world?", "answer": "vatican city"},
{"question": "Who painted 'The Starry Night'?", "answer": "van gogh"},
{"question": "What is the largest country by land area?", "answer": "russia"},
{"question": "What is the chemical symbol for silver?", "answer": "ag"},
{"question": "Who was the Queen of England for the longest time?", "answer": "elizabeth ii"},
{"question": "What is the process of a liquid turning into a gas?", "answer": "evaporation"},
{"question": "What is the name of the currency used in the United Kingdom?", "answer": "pound sterling"},
{"question": "What is the largest moon in our solar system?", "answer": "ganymede"},
{"question": "Who composed 'The Marriage of Figaro'?", "answer": "mozart"},
{"question": "What is the capital of China?", "answer": "beijing"},
{"question": "What is the primary function of red blood cells?", "answer": "oxygen transport"},
{"question": "What is the study of living organisms called?", "answer": "biology"},
{"question": "What is the chemical symbol for carbon?", "answer": "c"},
{"question": "Which mountain range runs along the western coast of South America?", "answer": "andes"},
{"question": "Who wrote 'The Odyssey'?", "answer": "homer"},
{"question": "What is the fastest animal on Earth?", "answer": "cheetah"},
{"question": "What is the freezing point of water in Fahrenheit?", "answer": "32"},
{"question": "What major historical event occurred in 1776?", "answer": "american independence"},
{"question": "What material is a pencil 'lead' made of?", "answer": "graphite"},
{"question": "Who discovered the laws of motion and universal gravitation?", "answer": "newton"},
{"question": "What is the name of the strait that separates Europe and Africa?", "answer": "gibraltar"},
{"question": "What is the chemical symbol for potassium?", "answer": "k"},
{"question": "Who was the leader of the Soviet Union during World War II?", "answer": "stalin"},
{"question": "What is the deepest point in the Earth's oceans?", "answer": "mariana trench"},
{"question": "What is the largest body of water in the world that is not an ocean?", "answer": "caspian sea"},
{"question": "Who composed 'Four Seasons'?", "answer": "vivaldi"},
{"question": "What is the process of turning a gas directly into a solid called?", "answer": "deposition"},
{"question": "Which country is home to the Kangaroo?", "answer": "australia"},
{"question": "What is the capital of Japan?", "answer": "tokyo"},
{"question": "What is the hardest material on the Mohs scale?", "answer": "diamond"},
{"question": "Who was the Ancient Greek mathematician known as the 'Father of Geometry'?", "answer": "euclid"},
{"question": "What is the chemical symbol for copper?", "answer": "cu"},
{"question": "What is the main component of Earth's core?", "answer": "iron and nickel"},
{"question": "What famous phrase did Julius Caesar reportedly say when crossing the Rubicon?", "answer": "alea iacta est"},
{"question": "What is the name of the ship that brought the Pilgrims to America?", "answer": "mayflower"},
{"question": "What is the unit of frequency?", "answer": "hertz"},
{"question": "What planet is known for its rings?", "answer": "saturn"},
{"question": "Who invented the printing press?", "answer": "gutenberg"},
{"question": "What is the most abundant element in the universe?", "answer": "hydrogen"},
{"question": "What is the capital of Mexico?", "answer": "mexico city"},
{"question": "Which classical composer was deaf?", "answer": "beethoven"},
{"question": "What is the name of the currency used in Russia?", "answer": "ruble"},
{"question": "What is the chemical formula for sulfuric acid?", "answer": "h2so4"},
{"question": "In which city is the Colosseum located?", "answer": "rome"},
{"question": "What are the two major components of a simple battery?", "answer": "anode and cathode"},
{"question": "What city is known as 'The Big Apple'?", "answer": "new york"},
{"question": "What is the capital of Brazil?", "answer": "brasilia"},
{"question": "Who wrote 'Pride and Prejudice'?", "answer": "austen"},
{"question": "What is the common name for sodium chloride?", "answer": "salt"},
{"question": "How many days are in a leap year?", "answer": "366"},
{"question": "Who was the Greek goddess of wisdom and war?", "answer": "athena"},
{"question": "What is the largest lake in Africa?", "answer": "victoria"},
{"question": "Who was the German philosopher known for 'Thus Spoke Zarathustra'?", "answer": "nietzsche"},
{"question": "What is the chemical symbol for calcium?", "answer": "ca"},
{"question": "What is the process of cell division in body cells called?", "answer": "mitosis"},
{"question": "What is the main type of rock found at the bottom of the ocean?", "answer": "sedimentary"},
{"question": "What is the capital of South Korea?", "answer": "seoul"},
{"question": "What famous landmark was built by the Emperor Shah Jahan?", "answer": "taj mahal"},
{"question": "What is the main pigment responsible for the green color in plants?", "answer": "chlorophyll"},
{"question": "Who composed the opera 'Carmen'?", "answer": "bizet"},
{"question": "What is the chemical symbol for tin?", "answer": "sn"},
{"question": "What is the name of the supercontinent that existed millions of years ago?", "answer": "pangea"},
{"question": "Who invented the steam engine?", "answer": "watt"},
{"question": "What is the capital of Egypt?", "answer": "cairo"},
{"question": "What is the primary language spoken in Mexico?", "answer": "spanish"},
{"question": "Who was the first female Prime Minister of the United Kingdom?", "answer": "thatcher"},
{"question": "What is the common name for the Aurora Borealis?", "answer": "northern lights"},
{"question": "What is the chemical symbol for lead?", "answer": "pb"},
{"question": "What is the unit of power?", "answer": "watt"},
{"question": "Which artist is famous for cutting his ear off?", "answer": "van gogh"},
{"question": "What is the largest country in South America?", "answer": "brazil"},
{"question": "What is the chemical formula for table sugar?", "answer": "c12h22o11"},
{"question": "Who was the author of 'Moby Dick'?", "answer": "melville"},
{"question": "What is the capital of Ireland?", "answer": "dublin"},
{"question": "What type of rock is formed from cooling magma?", "answer": "igneous"},
{"question": "Who was the first Roman Emperor?", "answer": "augustus"},
{"question": "What is the common name for the disease poliomyelitis?", "answer": "polio"},
{"question": "What is the name of the imaginary line that divides the Earth into North and South?", "answer": "equator"},
{"question": "What is the chemical symbol for mercury?", "answer": "hg"},
{"question": "What is the capital of New Zealand?", "answer": "wellington"},
{"question": "Who painted 'Guernica'?", "answer": "picasso"},
{"question": "What is the smallest ocean in the world?", "answer": "arctic"},
{"question": "What is the chemical symbol for neon?", "answer": "ne"},
{"question": "What famous document was signed in 1776?", "answer": "declaration of independence"},
{"question": "What are the three primary colors of light?", "answer": "red, green, blue"},
{"question": "Who is the author of 'The Canterbury Tales'?", "answer": "chaucer"},
{"question": "What is the capital of Greece?", "answer": "athens"},
{"question": "What is the process of generating energy from the sun?", "answer": "solar power"},
{"question": "What is the largest internal organ in the human body?", "answer": "liver"},
{"question": "What is the name of the Russian space agency?", "answer": "roscosmos"},
{"question": "What is the main language spoken in Portugal?", "answer": "portuguese"},
{"question": "Who was the longest-reigning British monarch?", "answer": "elizabeth ii"},
{"question": "What is the chemical symbol for sodium?", "answer": "na"},
{"question": "What is the unit of resistance?", "answer": "ohm"},
{"question": "In what country would you find Mount Kilimanjaro?", "answer": "tanzania"},
{"question": "Who wrote 'The Communist Manifesto'?", "answer": "marx"},
{"question": "What is the process of burning fuel called?", "answer": "combustion"},
{"question": "What is the capital of Sweden?", "answer": "stockholm"},
{"question": "What is the smallest planet in the solar system?", "answer": "mercury"},
{"question": "What element is denoted by the chemical symbol 'Sn'?", "answer": "tin"},
{"question": "Who designed the Statue of Liberty?", "answer": "bartholdi"},
{"question": "What is the currency of India?", "answer": "rupee"},
{"question": "What is the most widely spoken language in the world?", "answer": "mandarin chinese"},
{"question": "What is the name of the imaginary line that passes through Greenwich, England?", "answer": "prime meridian"},
{"question": "What famous ship was sunk by a German U-boat in 1915?", "answer": "lusitania"},
{"question": "What is the chemical symbol for chlorine?", "answer": "cl"},
{"question": "Who invented the World Wide Web?", "answer": "tim berners-lee"},
{"question": "What is the outer layer of the Earth called?", "answer": "crust"},
{"question": "What is the capital of Thailand?", "answer": "bangkok"},
{"question": "What type of energy is stored in a compressed spring?", "answer": "potential"},
{"question": "What is the deepest lake in the world?", "answer": "baikal"},
{"question": "Who painted 'The Scream'?", "answer": "munch"},
{"question": "What is the chemical symbol for zinc?", "answer": "zn"},
{"question": "What is the capital of Saudi Arabia?", "answer": "riyadh"},
{"question": "Who composed 'Clair de Lune'?", "answer": "debussy"},
{"question": "What is the name of the process that powers the sun?", "answer": "nuclear fusion"},
{"question": "What is the capital of Nigeria?", "answer": "abuja"},
{"question": "What is the chemical formula for methane?", "answer": "ch4"},
{"question": "What year did the Berlin Wall fall?", "answer": "1989"},
{"question": "Who wrote 'War and Peace'?", "answer": "tolstoy"},
{"question": "What is the main gas in Venus' atmosphere?", "answer": "carbon dioxide"},
{"question": "What is the capital of Argentina?", "answer": "buenos aires"},
{"question": "What is the pH level of neutral water?", "answer": "7"},
{"question": "Who developed the first successful printing press?", "answer": "gutenberg"}
]

# --- System Monitor ---
class SystemMonitor(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        # Find all GPUs
        try:
            self.gpus = Device.all()
            print(f"Monitoring {len(self.gpus)} GPU(s)")
        except (IndexError, Exception):
            print("No GPU found by nvitop. System monitoring will be limited.")
            self.gpus = []
        self.metrics = {
            'gpu_util': [],
            'vram_used_gb': [],
            'ram_percent': []
        }
        self.start_gpu = 0
        self.start_vram = 0
        self.initial_metrics_captured = False
        self.daemon = True

    def run(self):
        while not self.stop_event.is_set():
            if self.gpus:
                try:
                    # Track average across all GPUs
                    gpu_utils = []
                    vram_gbs = []

                    for gpu in self.gpus:
                        gpu_utils.append(gpu.gpu_utilization())
                        vram_gbs.append(gpu.memory_used() / 1024)

                    # Use average utilization and max VRAM (most conservative)
                    current_gpu_util = sum(gpu_utils) / len(gpu_utils)
                    current_vram_gb = max(vram_gbs)  # Max VRAM across all GPUs

                    if not self.initial_metrics_captured:
                        self.start_gpu = current_gpu_util
                        self.start_vram = current_vram_gb
                        self.initial_metrics_captured = True

                    self.metrics['gpu_util'].append(current_gpu_util)
                    self.metrics['vram_used_gb'].append(current_vram_gb)
                except Exception:
                    pass  # Ignore transient errors
            self.metrics['ram_percent'].append(psutil.virtual_memory().percent)
            time.sleep(1)

    def stop(self):
        self.stop_event.set()

    def get_summary(self):
        if not self.metrics['gpu_util']:
            avg_ram = sum(self.metrics['ram_percent']) / len(
                self.metrics['ram_percent']) if self.metrics['ram_percent'] else 0
            return {'peak_gpu': 0, 'avg_gpu': 0, 'peak_vram': 0, 'avg_vram': 0, 'avg_ram': round(
                avg_ram, 2), 'start_gpu': self.start_gpu, 'start_vram': self.start_vram}

        peak_gpu = max(self.metrics['gpu_util'])
        avg_gpu = sum(self.metrics['gpu_util']) / len(self.metrics['gpu_util'])
        peak_vram = max(self.metrics['vram_used_gb'])
        avg_vram = sum(self.metrics['vram_used_gb']) / \
            len(self.metrics['vram_used_gb'])
        avg_ram = sum(self.metrics['ram_percent']) / \
            len(self.metrics['ram_percent'])
        return {'peak_gpu': round(peak_gpu, 2), 'avg_gpu': round(avg_gpu, 2), 'peak_vram': round(peak_vram, 2), 'avg_vram': round(
            avg_vram, 2), 'avg_ram': round(avg_ram, 2), 'start_gpu': round(self.start_gpu, 2), 'start_vram': round(self.start_vram, 2)}


# --- CSV Logging & Test Logic ---
# ... (no changes to logging or test worker functions)
csv_lock = threading.Lock()


def initialize_csv():
    # Ensure results directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)

    if not os.path.exists(OUTPUT_CSV_FILE):
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp",
                             "Model",
                             "Concurrent_Threads",
                             "Thread_id",
                             "Test_Type",
                             "Loop_Num",
                             "Needle_Position",
                             "Question",
                             "Expected_Answer",
                             "Model_Output",
                             "Execution_Time_s",
                             "Accuracy","context_preview","context_word_count"])


def log_metadata(model_id, num_threads, summary):
    with csv_lock:
        with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            f.write("\n# --- METADATA --- \n")
            f.write(f"# Model: {model_id}\n")
            f.write(f"# Concurrent_Threads: {num_threads}\n")
            f.write(
                f"""# Test_Start_Time: {
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n""")
            f.write(f"# Starting_GPU_Utilization(%): {summary['start_gpu']}\n")
            f.write(f"# Starting_VRAM_Usage(GB): {summary['start_vram']}\n")
            f.write(f"# Peak_GPU_Utilization(%): {summary['peak_gpu']}\n")
            f.write(f"# Average_GPU_Utilization(%): {summary['avg_gpu']}\n")
            f.write(f"# Peak_VRAM_Usage(GB): {summary['peak_vram']}\n")
            f.write(f"# Average_VRAM_Usage(GB): {summary['avg_vram']}\n")
            f.write(f"# Average_System_RAM_Usage(%): {summary['avg_ram']}\n")
            f.write("# ---\n")
            f.flush()  # Explicitly flush to disk to prevent data loss on OOM
            os.fsync(f.fileno())  # Force OS to write to disk


def log_result(model_id, num_threads,Thread_id,test_type, loop_num,
               position, question, expected, output, exec_time, accuracy, context_preview,context_word_count):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [
        timestamp,
        model_id,
        num_threads,
        Thread_id,
        test_type,
        loop_num,
        position,
        question,
        expected,
        output,
        round(
            exec_time,
            2),
        accuracy,context_preview[:500],context_word_count]
    with csv_lock:
        with open(OUTPUT_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()  # Explicitly flush to disk to prevent data loss on OOM
            os.fsync(f.fileno())  # Force OS to write to disk

# --- Test Generation & Evaluation ---


def check_accuracy(model_output, expected_answer):
    return "Pass" if expected_answer.lower() in model_output.lower() else "Fail"


def create_needle_haystack_prompt(
        haystack_text, position_name, position_ratio):
    haystack_words = haystack_text.split()
    needle = "The secret code to activate the system is 'BluePanda-42'."
    expected_answer = "BluePanda-42"
    question = "Based on the document, what is the secret code to activate the system?"
    insertion_point = int(len(haystack_words) * position_ratio)
    haystack_words.insert(insertion_point, needle)
    haystack_with_needle = " ".join(haystack_words)
    prompt = f"""Please analyze the following document carefully and answer the question that follows.\n\nDOCUMENT:\n---\n{
        haystack_with_needle}\n---\n\nQUESTION:\n{question}"""
    return prompt, question, expected_answer

# --- Main Test Logic ---


def test_worker(client, model_id, num_threads,Thread_id,
                context_length, full_haystack_text,max_words):
    try:
        # max_words = int(context_length / 1.3)
        # max_words = 4096
        max_words = max_words
        haystack_words = full_haystack_text.split()
        if len(haystack_words) > max_words:
            haystack_words = haystack_words[:max_words]
        truncated_haystack = " ".join(haystack_words)
        context_word_count = len(truncated_haystack.split())
        context_preview = f"{truncated_haystack[:200]}... ({context_word_count} words)"

        for i in range(1, SIMPLE_BASELINE_LOOPS + 1):
            for qa_pair in GENERAL_KNOWLEDGE_QUESTIONS:
                q, ans = qa_pair["question"], qa_pair["answer"]
                prompt = f"""Based on your general knowledge ONLY, and strictly ignoring the long text below, please answer the following question.
IMPORTANT:
- You must NOT use, quote, or rely on the DOCUMENT below in any way.
- The DOCUMENT is irrelevant and must be completely ignored.
DOCUMENT (DO NOT USE):
{truncated_haystack}
QUESTION:{q}
ANSWER (based on general knowledge only)"""
                start_time = time.time()
                print(model_id)
                completion = client.completions.create(
                model=model_id, prompt=prompt, max_tokens=100, temperature=0.0)
                print(completion)
                model_output = completion.choices[0].text.strip()
                exec_time = time.time() - start_time
                accuracy = check_accuracy(model_output, ans)
                log_result(
            model_id,
            num_threads,
            Thread_id,
            "Simple_Baseline",
            i,
            "N/A",
            q,
            ans,
            model_output,
            exec_time,
            accuracy,context_preview,context_word_count)
        for loop in range(1, NEEDLE_HAYSTACK_LOOPS + 1):
            needle_positions = [("start", 0.05), ("middle", 0.5), ("end", 0.95)]
            for pos_name, pos_ratio in needle_positions:
                #pos_name, pos_ratio = random.choice(needle_positions)
                prompt, q, ans = create_needle_haystack_prompt(
                truncated_haystack, pos_name, pos_ratio)
                start_time = time.time()
                print(model_id)
                completion = client.completions.create(
                    model=model_id, prompt=prompt, max_tokens=100, temperature=0.0)
                print(completion)
                model_output = completion.choices[0].text.strip()
                exec_time = time.time() - start_time
                accuracy = check_accuracy(model_output, ans)
                log_result(
            model_id,
            num_threads,
            Thread_id,
            "Needle_Haystack",
            loop,
            pos_name,
            q,
            ans,
            model_output,
            exec_time,
            accuracy,context_preview,context_word_count)

    except MemoryError as e:
        error_msg = f"OOM Error in thread: {str(e)}"
        print(f"ERROR: {error_msg}")
        # Try to log the error before process dies
        try:
            log_result(
                model_id,
                num_threads,
                Thread_id,
                "ERROR",
                "OOM",
                "N/A",
                "Out of Memory",
                "N/A",
                error_msg,
                0,
                "Fail","",0)
        except BaseException:
            pass  
        raise  # Re-raise to propagate
    except Exception as e:
        error_msg = f"Error in thread: {str(e)}"
        print(f"ERROR: {error_msg}")
        # Log the error to CSV
        try:
            log_result(model_id, num_threads, "ERROR", "Exception",
                       "N/A", str(type(e).__name__), "N/A", error_msg, 0, "Fail","",0)
        except BaseException:
            pass  # If logging fails, continue


def main():
    """Main function to orchestrate the entire testing process against a vLLM server."""
    initialize_csv()

    print(
        f"""\n{
            '=' *
            20} Starting tests for model: {MODEL_BEING_TESTED} {
            '=' *
            20}""")
    print(
        f"Ensure this model is currently running on the vLLM server at {VLLM_HOST}")

    try:
        with open(HAYSTACK_FILE, 'r', encoding='utf-8') as f:
            haystack_text = f.read()
    except FileNotFoundError:
        print(f"""Error: The haystack file '{
              HAYSTACK_FILE}' was not found. Please create it.""")
        return

    client = OpenAI(base_url=VLLM_HOST, api_key=API_KEY)
    context_length_for_test =MODEL_MAX_CONTEXT
    print(f"""Model max context: {MODEL_MAX_CONTEXT}, using ~{
          context_length_for_test} tokens for tests.""")
    max_words=MAX_WORDS_REQUIRED

    # --- NEW: DYNAMIC THREADING LOOP ---
    num_threads = STARTING_THREADS
    while True:
        if num_threads > MAX_THREADS_SAFETY_LIMIT:
            print(f"""Reached safety limit of {
                  MAX_THREADS_SAFETY_LIMIT} threads. Stopping test.""")
            break

        print(f"\n--- Running test with {num_threads} concurrent threads ---")

        monitor = SystemMonitor()
        monitor.start()

        threads = []
        for Thread_id in range(num_threads):
            thread = threading.Thread(
                target=test_worker,
                args=(
                    client,
                    MODEL_BEING_TESTED,
                    num_threads,
                    Thread_id,
                    context_length_for_test,
                    haystack_text,max_words))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        monitor.stop()
        monitor.join()
        summary = monitor.get_summary()

        log_metadata(MODEL_BEING_TESTED, num_threads, summary)
        print("Run complete. Metrics logged.")
        print(
            f"""Peak GPU: {
                summary['peak_gpu']}% | Peak VRAM: {
                summary['peak_vram']:.2f}GB | Avg VRAM: {
                summary['avg_vram']:.2f}GB""")

        # Check if we should stop (GPU utilization OR VRAM threshold)
        if summary['peak_gpu'] > GPU_UTILIZATION_THRESHOLD:
            print(f"""GPU utilization reached {summary['peak_gpu']}% (>{
                  GPU_UTILIZATION_THRESHOLD}%). Concluding tests for this model.""")
            break

        num_threads += THREAD_INCREMENT

        print(f"Entering {COOLING_PERIOD_S}s cooling period...")
        time.sleep(COOLING_PERIOD_S)

    print(
        f"""\n{
            '=' *
            20} Finished all configured tests for model: {MODEL_BEING_TESTED} {
            '=' *
            20}""")


if __name__ == "__main__":
    main()

