import sys
import keyboard
from generate import init_model, load_artifacts, predict

class AutoCompleteEngine:

    def __init__(self):

        self.buffer = ""
        self.suggestion = ""
        self.config, self.stoi, self.itos, self.C = load_artifacts()
        self.model = init_model(self.config, self.C)
        self.gen = None
        self.init_gen()

    def init_gen(self):

        self.gen = predict(
            self.model,
            self.C,
            self.itos,
            self.stoi,
            self.config,
        )
        next(self.gen)

    def update_suggestion(self):

        if not self.buffer:
            self.suggestion = ""
            return

        current_word = self.buffer.split()[-1] if self.buffer and not self.buffer.endswith(' ') else ""

        if current_word and len(current_word) > 0:
            try:
                self.suggestion = self.gen.send(current_word[-1]) or ""

            except StopIteration:
                self.init_gen()
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

        while True:
            
            event = keyboard.read_event()

            if event.event_type != keyboard.KEY_DOWN:
                continue

            if event.name == "esc":
                break

            if event.name == "backspace":
                if self.buffer:
                    self.buffer = self.buffer[:-1]

                    self.init_gen()
                    self.gen.send(('set_text', self.buffer))
                    self.suggestion = ""

            elif event.name == "space":
                if self.suggestion:
                    self.buffer += self.suggestion
                    self.suggestion = ""
                self.buffer += " "

                self.gen.send(('set_text', self.buffer))

            elif event.name == "tab":
                if self.suggestion:
                    self.buffer += self.suggestion
                    self.gen.send(('set_text', self.buffer))
                    self.suggestion = ""

            elif len(event.name) == 1:
                self.buffer += event.name
                self.update_suggestion()

            self.display()

engine = AutoCompleteEngine()
engine.run()
