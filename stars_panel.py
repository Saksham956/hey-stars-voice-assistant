"""Main-thread Windows panel for Core v2. Workers only enqueue log events."""
import ctypes
import logging
import queue
import threading

class PanelState:
    def __init__(self):
        self.status='Ready — say Hey Jarvis'
        self.heard=''
        self.answer=''
        self.latency=''
        self.tasks='Click Show tasks to read your saved tasks.'
    def consume(self, text):
        if text.startswith('LOCAL TASKS:'):
            self.tasks=text.split(':',1)[1].strip()
        elif text.startswith('Listening for command'):
            self.status='Listening…'
            self.heard=self.answer=self.latency=''
        elif text.startswith('Sending one request'):
            self.status='Thinking…'
        elif text.startswith(('YOU (Gemini transcript):','YOU (local transcript):')):
            self.heard=text.split(':',1)[1].strip()
        elif text.startswith('STARS (queued for speech):'):
            self.answer=(self.answer+' '+text.split(':',1)[1].strip()).strip()[-12000:]
            self.status='Preparing voice…'
        elif text.startswith('LOCAL RESULT:'):
            self.answer=text.split(':',1)[1].strip()
            self.status='Ready - say Hey Jarvis'
        elif text.startswith('TTFA generation='):
            self.status='Speaking…'
            self.latency=text.split('end-of-speech-to-first-PCM=',1)[-1]
        elif text.startswith(('Gemini request failed','Cloud response timed out','Request blocked','No speech detected','Request cancelled','Microphone stopped','Local TTS failed')):
            self.status=text

class PanelHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events=queue.Queue(128)
    def emit(self, record):
        try: self.events.put_nowait(record.getMessage())
        except queue.Full: pass

class StarsPanel:
    def __init__(self, stop, epoch, output, mic, turns, local_router=None):
        import tkinter as tk
        self.tk=tk
        self.stop,self.epoch,self.output,self.mic,self.turns=stop,epoch,output,mic,turns
        self.local_router=local_router
        self.tasks_busy=threading.Event()
        self.state=PanelState()
        self.handler=PanelHandler()
        self.root=tk.Tk()
        self.root.title('STARS - Hey Jarvis')
        self.root.geometry('480x430')
        self.root.configure(bg='#111827')
        self.root.protocol('WM_DELETE_WINDOW',self.root.iconify)
        self.status=tk.StringVar(value=self.state.status)
        self.heard=tk.StringVar(value='Your question appears here.')
        self.latency=tk.StringVar()
        self.tasks=tk.StringVar(value=self.state.tasks)
        tk.Label(self.root,text='✦ STARS',font=('Segoe UI',20,'bold'),fg='#7dd3fc',bg='#111827').pack(pady=(14,4))
        tk.Label(self.root,textvariable=self.status,wraplength=420,fg='white',bg='#111827').pack(pady=4)
        tk.Label(self.root,textvariable=self.heard,wraplength=420,justify='left',fg='#a5b4fc',bg='#111827').pack(padx=16,pady=8)
        self.reply=tk.Text(self.root,height=6,wrap='word',font=('Segoe UI',11),bg='#1f2937',fg='white',relief='flat')
        self.reply.pack(fill='both',expand=True,padx=16)
        self.reply.configure(state='disabled')
        tk.Label(self.root,textvariable=self.latency,fg='#9ca3af',bg='#111827').pack(pady=4)
        tk.Label(self.root,textvariable=self.tasks,wraplength=440,justify='left',fg='#d1fae5',bg='#111827').pack(padx=16,pady=6)
        controls=tk.Frame(self.root,bg='#111827');controls.pack(pady=8)
        tk.Button(controls,text='Show tasks',command=self.show_tasks,
                  state='normal' if local_router is not None else 'disabled').pack(side='left',padx=4)
        self.pause_button=tk.Button(controls,text='Pause listening',command=self.pause)
        self.pause_button.pack(side='left',padx=4)
        tk.Button(controls,text='Minimize',command=self.root.iconify).pack(side='left',padx=4)
        tk.Button(controls,text='Quit STARS',command=stop.set).pack(side='left',padx=4)
        self.root.update_idletasks()
        # Keep our own panel out of cached screenshots where Windows supports it.
        user32=ctypes.windll.user32
        user32.GetAncestor.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        user32.GetAncestor.restype=ctypes.c_void_p
        user32.SetWindowDisplayAffinity.argtypes=[ctypes.c_void_p,ctypes.c_uint]
        hwnd=user32.GetAncestor(self.root.winfo_id(),2)
        self.capture_excluded=bool(user32.SetWindowDisplayAffinity(hwnd,0x11))
        logging.getLogger('stars').addHandler(self.handler)
    def show_tasks(self):
        if self.local_router is None or self.tasks_busy.is_set(): return
        self.tasks_busy.set()
        def read_tasks():
            from local_commands import Intent
            try:
                message=self.local_router.execute(Intent('tasks')).message
            except Exception as exc:
                message='Could not read saved tasks ('+type(exc).__name__+').'
            try:
                # Worker communicates with Tk only through the main-thread queue.
                if not self.stop.is_set():
                    self.handler.events.put_nowait('LOCAL TASKS: '+message)
            except queue.Full:
                pass
            finally:
                self.tasks_busy.clear()
        threading.Thread(target=read_tasks,name='task-reader',daemon=True).start()
    def pause(self):
        paused=not self.mic.paused.is_set()
        if paused:
            self.mic.paused.set()
            self.epoch.advance()
            self.output.interrupt()
            from hey_stars import clear_queue
            clear_queue(self.turns)
        else: self.mic.paused.clear()
        self.pause_button.configure(text='Resume listening' if paused else 'Pause listening')
        self.status.set('Paused — no commands accepted' if paused else 'Ready — say Hey Jarvis')
    def tick(self):
        while True:
            try: text=self.handler.events.get_nowait()
            except queue.Empty: break
            if text.startswith('LOCAL TASKS:'):
                self.state.consume(text)
                self.tasks.set(self.state.tasks)
                continue
            if self.mic.paused.is_set(): continue
            self.state.consume(text)
            if text.startswith('Listening for command'):
                self.root.deiconify()
                self.root.lift()
            self.status.set(self.state.status)
            self.heard.set('You: '+self.state.heard if self.state.heard else '')
            self.latency.set('First audio: '+self.state.latency if self.state.latency else '')
            content=self.reply.get('1.0','end-1c')
            if content!=self.state.answer:
                self.reply.configure(state='normal')
                self.reply.delete('1.0','end');self.reply.insert('end',self.state.answer)
                self.reply.configure(state='disabled')
        if self.state.status=='Speaking…' and not self.output.synthesizing.is_set() and self.output.sentences.empty() and self.output.audio.empty() and self.output.current is None:
            self.state.status='Ready — say Hey Jarvis'
            self.status.set(self.state.status)
        self.root.update()
    def close(self):
        logging.getLogger('stars').removeHandler(self.handler)
        self.root.destroy()
