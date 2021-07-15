import json
import jsonlines
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
#from pprint import pprint
"""
preprocess for chit-chat dataset
"""

def main(args):
    dialogues_list = list(args.data_path.glob('*.json'))
    dialogues_list.sort()
    #pprint(dialogues_list)
    processed_data = []
    print("*****-----*****----- preprocessing -----*****-----*****")
    progress_bar = tqdm(total=len(dialogues_list))
    processed_data = []
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            history = ""
            belief = ""
            for idx,turn in enumerate(dialogue['turns']):
                # USER turn: goal collect state
                if turn['speaker'] == "USER":
                    history += " <|user|> "+ turn['utterance'].strip()
                    state = []
                    belief = ""
                    if len(turn['frames']) == 0:
                        continue
                    domain = turn['frames'][0]['service'].split('_')[0].lower()
                    for k, v in turn['frames'][0]['state']['slot_values'].items():
                        state.append([domain, k, v])
                    state.sort(key = lambda x: x[0] + " " + x[1])
                    for s in state:
                        s[2].sort()
                        s[2] = s[2][0]
                    
                    state = [s[0] + " " + s[1] + " " + s[2] for s in state]
                    belief = "<|belief|> " + ", ".join(state) + " <|endofbelief|> "
                # SYSTEM turn: collect action, collect chit-chat, collect response, replace utterance task slot with special slot 
                else:
                    if len(turn['frames']) == 0:
                        break
                    utterance = turn["utterance"].strip()
                    domain = turn['frames'][0]['service'].split('_')[0].lower()
                    # *****----- collect action ------*****
                    action = deepcopy(turn['frames'][0]['actions'])
                    action.sort(key = lambda x : x["act"])
                    action = [domain + " " + x["act"].lower() + " " + x["slot"] for x in action]
                    # *****----- collect action ------*****
                    # *****----- replace utterance ------*****
                    replace_slots = deepcopy(turn['frames'][0]['actions'])
                    replace_slots = [x for x in replace_slots if len(x['values']) > 0]
                    is_replaced = set()
                    for replace_slot in replace_slots:
                        domain_slot = domain + "_" + replace_slot['slot']
                        if domain_slot in is_replaced:
                            continue
                        for origin_slot in replace_slot['values']:
                            utterance = utterance.replace(origin_slot,"["+domain_slot+"]")
                        is_replaced.add(domain_slot)
                    # *****----- replace utterance ------*****
                    # *****----- collect chit-chat and generate sequence input ------*****
                    if 'beginning' in turn:
                        for candidate in turn['beginning']:
                            if candidate['label'] == "good":
                                seq = "<|context|>" + history + " <|endofcontext|> " + belief + "<|action|> " + "chit-chat in beginning, " +", ".join(action) + " <|endofaction|> " + "<|chitchat|> " + candidate['candidate'] + " <|endofchitchat|> " + "<|response|> " + utterance + " <|endofresponse|>"
                                processed_data.append({"text":"<|endoftext|>"+seq+"<|endoftext|>"})
                    if 'end' in turn:
                        for candidate in turn['end']:
                            if candidate['label'] == "good":
                                seq = "<|context|>" + history + " <|endofcontext|> " + belief + "<|action|> " + "chit-chat in end, " + ", ".join(action) + " <|endofaction|> " + "<|chitchat|> " + candidate['candidate'] + " <|endofchitchat|> " + "<|response|> " + utterance + " <|endofresponse|>"
                                processed_data.append({"text":"<|endoftext|>"+seq+"<|endoftext|>"})
                    # no chit-chat case
                    if ('beginning' not in turn and 'end' not in turn) or (len(turn['beginning']) == 0 and len(turn['end']) == 0):
                        seq = "<|context|>" + history + " <|endofcontext|> " + belief + "<|action|> " +", ".join(action) + " <|endofaction|> " + "<|chitchat|> " + " <|endofchitchat|> " + "<|response|> " + utterance + " <|endofresponse|>"
                        processed_data.append({"text":"<|endoftext|>"+seq+"<|endoftext|>"})
                    # *****----- collect chit-chat and generate sequence input ------*****
                    # *****----- record history ------*****
                    history += " <|system|> " + utterance
                    # *****----- record history ------*****
        progress_bar.update(1)
    
    
    output_name = os.path.join(args.output_dir,args.data_name+".jsonl")
    output_json = os.path.join(args.output_dir,args.data_name+".json")
    with jsonlines.open(output_name,mode="w") as writer:
        for datapoint in processed_data:
            writer.write(datapoint)
    json.dump(processed_data,open(output_json,'w'),indent=4)
    print("\n")
    print("*****-----*****----- Completed preprocess -----*****-----*****")
    return

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # *****-----*****----- arguments -----*****-----*****
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="./cache/"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="./data-0614/data-0614/train"
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="train_with_state_chitchat"
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )

    # *****-----*****----- arguments -----*****-----*****
    args = parser.parse_args()
    return args

def test_preprocess(args):
    dialogues_list = list(args.data_path.glob('*.json'))
    dialogues_list.sort()
    #pprint(dialogues_list)
    processed_data = []
    print("*****-----*****----- preprocessing -----*****-----*****")
    progress_bar = tqdm(total=len(dialogues_list))
    processed_data = []
    for dialogues_path in dialogues_list:
        with open(dialogues_path) as f:
            dialogues = json.load(f)
        for dialogue in dialogues:
            history = ""
            for idx,turn in enumerate(dialogue['turns']):
                speaker = " <|user|> " if turn['speaker']=="USER" else " <|system|> "
                history += speaker + turn['utterance'].strip()
                if speaker == " <|user|> ":
                    processed_data.append({'text':"<|context|>"+history+" <|endofcontext|>"})
        progress_bar.update(1)
    output_name = os.path.join(args.output_dir,args.data_name+".jsonl")
    output_json = os.path.join(args.output_dir,args.data_name+".json")
    with jsonlines.open(output_name,mode="w") as writer:
        for datapoint in processed_data:
            writer.write(datapoint)
    json.dump(processed_data,open(output_json,'w'),indent=4)
    print("\n")
    print("*****-----*****----- Completed preprocess -----*****-----*****")
    return

if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.do_test:
        test_preprocess(args)
    else:
        main(args)