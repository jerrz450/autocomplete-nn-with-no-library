import sys
import keyboard
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate import init_model, load_artifacts, predict

class AutoCompleteEngine:

    def __init__(self):

        self.buffer = ""
        self.suggestion = ""
        self.config, self.stoi, self.itos, self.C = load_artifacts()
        self.model = init_model(self.config, self.C)
        self.gen = None
        self.last_update = 0
        self.update_delay = 0.05
        self.init_gen()

    def init_gen(self):

        if self.gen:
            try:
                self.gen.close()

            except:
                pass 

            del self.gen

        self.gen = predict(
            self.model,
            self.C,
            self.itos,
            self.stoi,
            self.config,
        )
        next(self.gen)

    def update_suggestion(self):

        current_time = time.time()
        
        if current_time - self.last_update < self.update_delay:
            return
        
        self.last_update = current_time

        if not self.buffer or self.buffer.endswith(' '):
            self.suggestion = ""
            return

        current_word = self.buffer.split()[-1]

        if current_word and len(current_word) > 0:
            try:
                self.suggestion = self.gen.send(current_word[-1]) or ""

            except (StopIteration, Exception):

                self.init_gen()

                if self.buffer:
                    
                    try:
                        self.gen.send(('set_text', self.buffer))
                    except:
                        pass

                self.suggestion = ""
        else:
            self.suggestion = ""

    def display(self):

        sys.stdout.write("\r\033[K> " + self.buffer)
        
        if self.suggestion:
            sys.stdout.write("\033[90m" + self.suggestion + "\033[0m")

        sys.stdout.flush()

    def run(self):

        print("> ", end="", flush=True)
        last_event_time = {}

        while True:

            event = keyboard.read_event()

            if event.event_type != keyboard.KEY_DOWN:
                continue

            current_time = time.time()
            if event.name in last_event_time and current_time - last_event_time[event.name] < 0.01:
                continue
            last_event_time[event.name] = current_time

            if event.name == "esc":
                break

            if event.name == "backspace":

                if self.buffer:
                    self.buffer = self.buffer[:-1]
                    self.init_gen()

                    if self.buffer:
                        try:
                            self.gen.send(('set_text', self.buffer))
                        except:
                            pass
                    self.suggestion = ""

            elif event.name == "space":

                if self.suggestion:
                    self.buffer += self.suggestion
                    self.suggestion = ""

                self.buffer += " "
                if self.buffer:
                    try:
                        self.gen.send(('set_text', self.buffer))
                    except:
                        pass

            elif event.name == "tab":

                if self.suggestion:
                    self.buffer += self.suggestion

                    if self.buffer:
                        try:
                            self.gen.send(('set_text', self.buffer))
                        except:
                            pass
                    self.suggestion = ""

            elif len(event.name) == 1 and event.name.isprintable():
                self.buffer += event.name
                self.update_suggestion()

            self.display()

engine = AutoCompleteEngine()
engine.run()
