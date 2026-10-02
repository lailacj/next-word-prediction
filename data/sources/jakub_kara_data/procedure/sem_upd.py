from psychopy import core, visual, logging, event, gui, parallel, logging
from psychopy.hardware import keyboard
from random import random
import time
import socket
import sys
#import pyxid2 as pyxid



#monitor = 'lapek'
monitor = 'CABIN1'
EEG = 1
rr = 60.0 #refresh rate monitora

LOG_HOST = '192.17.53.41'
LOG_PORT = 17322


#przypisanie wersji zadania - moze byc jako argument w linii polecen, jak nie ma to defaultowo wersja 1
if len(sys.argv) < 2:
    VER = "1"
else:
    VER = sys.argv[1]

ver_num = int(VER[0])


ms100 = rr / 10.0

#devices = pyxid.get_xid_devices()
#cedrus = devices[0]

addr = (LOG_HOST, LOG_PORT)
UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_udp(msg):
    UDPSock.sendto(msg.encode(), addr)


def exit_proc():
    core.quit()
event.globalKeys.clear()
event.globalKeys.add(key='q', modifiers=['ctrl', 'alt'], func=exit_proc)

def sendEEGTrig(trigger):
    if not trigger or not EEG:
        return
    port.setData(trigger)
    core.wait(0.0005)
    port.setData(0)

def show_seq(words, times, triggers, tolog=""):
    assert len(words) == len(times)
    current_thr = 0
    current_item = -1
    for frameN in range(int(sum(times))):
        if frameN == current_thr:
            current_item += 1
            current_thr += int(times[current_item])
            text.text = words[current_item]
            
            if (words[current_item] != "" or triggers[current_item] != 0):
                datetime = time.strftime('%Y-%m-%d %H:%M:%S')
                msg = "%s\t%s\t%d\t%s" % (datetime, tolog, triggers[current_item], words[current_item])
                win.logOnFlip(level=logging.EXP, msg=msg)
                
            win.flip()
            sendEEGTrig(triggers[current_item])
        else:
            win.flip()


def wait_for_space():
    event.clearEvents()
    kb.getKeys(clear=True)
    while 1:
        keypresses = kb.getKeys(['space'])
        if (len(keypresses)):
            break
        win.flip()
    return
    



all_items = { "train": [], "stage1": [], "stage2": [], "stage3": [] }
stages = [ "train", "stage1", "stage2", "stage3" ]
with open("2019.12.03 - sem upd v%s.txt" % (VER), "r") as f:
    col_names = f.readline().strip().split("\t")
    ix = 0
    stage = 1
    stage_name = "stage1"
    training = 1
    
    for row in f:
        ix += 1
        row_dict = {}
        parts = row.strip().split("\t")
        for cx in range(len(parts)):
            row_dict[col_names[cx]] = parts[cx]
        
        words = row_dict['context'].split(" ")
        wx = len(words) - 2
        mark = 0
        
        
#        while (wx >= 0):
#            if words[wx].lower() in [ 'a', 'an', 'the', 'too' ] or words[wx+1] in [ "not" ]:
#                words[wx] = words[wx] + ' ' + words[wx+1]
#                del words[wx+1]
#            wx -= 1
        

        row_dict['context'] = words
        row_dict['ItemExpID'] = int(row_dict['ItemExpID'])
        row_dict['cond'] = int(row_dict['cond'])
        
        len_words = len(words)
        
        if (row_dict['typ'] == "train"):
            base = 0
            triggers = [ base ] * (len_words + 2) # +2 bc adj & noun

        else:
            base = (row_dict['cond'] << 4) #condition number, 2 bits
            base += ((row_dict['ItemExpID'] & 3) << 6) #last two bits of internal item ID number
            triggers = [ base ] * (len_words + 2) # +2 bc adj & noun
            for wx in range (len_words):
                trigger = min(len_words + 2 - wx, 15) #words at pos 16+ -> pos=15; +2 bc room for adj&noun
                triggers[wx] += trigger

        row_dict['trigger_base'] = base
        row_dict['triggers'] = triggers
                
        if (training):
            if (row_dict["typ"] == "train"):
                all_items["train"].append(row_dict)
                continue
            else:
                training = 0
                ix = 1
        if (ix == 81):
            ix = 1
            stage += 1
        all_items[stages[stage]].append(row_dict)

log = logging.LogFile("exp sem_upd v%s.log" % (VER), level=logging.EXP, filemode='a')

dlg_dict = { 'Subject ID': ''}
dlg = gui.DlgFromDict(dlg_dict, title='New session')
if not dlg.OK:
    core.quit()



win = visual.Window(size=[1280, 1024], monitor=monitor, fullscr=True, units='deg', viewPos=None, color=[-0.2, -0.2, -0.2])
win.recordFrameIntervals = True
win.refreshThreshold = 1/rr + 0.004
logging.console.setLevel(logging.WARNING)

text = visual.TextStim(win, text='', alignText='center', height=0.8, units='deg', autoLog=False)
text.autoDraw = True
text.draw()


kb = keyboard.Keyboard()
win.mouseVisible = False
if EEG:
    port = parallel.ParallelPort(address=0x0378)

#bity:
#cond = 2
#pos = 4
#item = 2


text.text = "v%s" % (VER)
wait_for_space()




#kb.start()
#kb.clock.reset()

countdown = 0
for stage in stages:
    countdown += len(all_items[stage])

for stage in stages:
    items = all_items[stage]
    if (stage == "train"):
        text.text = "Training session.\n\nPlease press SPACE to proceed."
    elif (stage == "stage1"):
        text.text = "End of the training session.\n\nPlease wait for the experimenter."
        wait_for_space()
        text.text = "The main part of the study.\n\nPlease press SPACE to proceed."
    else:
        text.text = "Time for a break.\n\nPlease press SPACE when you are ready."
    wait_for_space()
    
    for item in items:
        countdown -= 1
        send_udp("%d" % (countdown))
        show_seq([""], [10*ms100], [0])
        text.text = "---"
        wait_for_space() 
        words = [ "", "+++", "" ]
        times = [ 10*ms100, 5*ms100, round((4.0 + random()*4.0) * ms100) ]
        triggers = [0, item['trigger_base'], 0]

        for wx in range(len(item['context'])):
            words += [ item['context'][wx], "" ]
            times += [ 3*ms100, 2*ms100 ]
            triggers += [ item['triggers'][wx], 0 ]
        words += [ item['adj'], "", item['noun'] + ".", "", "" ]
        times += [ 3*ms100, 2*ms100, 3*ms100, 10*ms100, 2*ms100 ]
        if stage == "train":
            triggers += [ 0 ] * 5
        else:
            triggers += [ item['trigger_base']+2, 0, item['trigger_base']+1, 0, item['ItemExpID'] ]
        
        show_seq(words, times, triggers, "%s\t%s\t%d" % (dlg_dict['Subject ID'], item['Item'], countdown))
        logging.flush()

text.text = "This is the end of this part of the study.\n\nPlease wait for the experimenter."
wait_for_space()

print('Overall, %i frames were dropped.' % win.nDroppedFrames)


core.quit()
