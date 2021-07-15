
import json
import jsonlines
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm
from state_to_csv import write_csv
import pickle
#from pprint import pprint
"""
postprocess for chit-chat dataset
"""
def parse_args() -> Namespace:
    parser = ArgumentParser()
    # *****-----*****----- arguments -----*****-----*****
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./test_seen_2_state.json"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default="./prediction/test_seen_2.json"
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default="./data/data/test_seen/"
    )
    parser.add_argument(
        "--chit_chat_state",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--do_chit_chat",
        action="store_true",
    )
    parser.add_argument(
        "--do_all",
        action="store_true",
    )

    # *****-----*****----- arguments -----*****-----*****
    args = parser.parse_args()
    return args

def main(args):
    error_count = 0
    is_error = False
    with open(args.data_path) as f:
        pred_data = json.load(f)
    output = []
    for pred in pred_data:
        text = pred['text']
        if text.find("<|action|>") != -1 and text.find("<|endofaction|>") != -1 and text.count("<|action|>") == 1 and text.count("<|endofaction|>") == 1:
            action = text[text.find("<|action|>"):text.find("<|endofaction|>")]
            action = action.replace("<|action|>","").split(',')
            action = [x.strip() for x in action]
        else:
            is_error = True
            action = []
        if text.find("<|belief|>") != -1 and text.find("<|endofbelief|>") != -1 and text.count("<|belief|>") == 1 and text.count("<|endofbelief|>") == 1:
            belief = text[text.find("<|belief|>"):text.find("<|endofbelief|>")]
            belief = belief.replace("<|belief|>","").split(',')
            belief = [x.strip() for x in belief]
        else:
            is_error = True
            belief = []
        new_belief = []
        for state in belief:
            s = state.split()
            if len(s) < 3:
                is_error = True
            elif (True in [s[x] == s[x+1] for x in range(0,len(s)-1)]):
                is_error = True
            else:
                new_belief.append(state)
        response = text[text.find("<|response|>"):text.find("<|endofresponse|>") if text.find("<|endofresponse|>") != -1 else None].replace("<|response|>","").strip()
        beginning = ""
        end = ""
        if not args.chit_chat_state:
            if "chit-chat" in action:
                if "chit-chat" == action[0]:
                    beginning = response.replace('?','.').replace('!','.').split('.')[0]
                elif "chit-chat" == action[-1]:
                    end = response.replace('?','.').replace('!','.').split('.')[-1]
        else:
            if "chit-chat in beginning" in action:
                beginning = text[text.find("<|chitchat|>"):text.find("<|endofchitchat|>")].replace("<|chitchat|>","").strip()
            elif "chit-chat in end" in action:
                end = text[text.find("<|chitchat|>"):text.find("<|endofchitchat|>")].replace("<|chitchat|>","").strip()
        output.append({
            'belief':new_belief,
            'action':action,
            'response':response,
            'beginning':beginning,
            'end':end
        })
        if is_error:
            error_count += 1
            is_error = False
    print("Failure probrability: ",error_count/len(pred_data))
    if args.do_all:
        dialogues_list = list(args.test_dir.glob('*.json'))
        dialogues_list.sort()
        processed_data = {}
        print("*****-----*****----- post processing all -----*****-----*****")
        for dialogues_path in dialogues_list:
            with open(dialogues_path) as f:
                dialogues = json.load(f)
            for dialogue in dialogues:
                processed_data[dialogue['dialogue_id']] = {}
                for idx,turn in enumerate(dialogue['turns']):
                    if turn['speaker'] == "SYSTEM":
                        assert output
                        output_data = output.pop(0)
                        processed_data[dialogue['dialogue_id']][turn['turn_id']] = {
                            'beginning':output_data['beginning'],
                            'end':output_data['end'],
                            'response':output_data['response'],
                            'belief':output_data['belief'],
                            'action':output_data['action']
                        }
        assert not output
        json.dump(processed_data,open(args.output_path,"w"),indent=2)
    elif args.do_chit_chat:
        dialogues_list = list(args.test_dir.glob('*.json'))
        dialogues_list.sort()
        processed_data = {}
        print("*****-----*****----- post processing chit chat -----*****-----*****")
        for dialogues_path in dialogues_list:
            with open(dialogues_path) as f:
                dialogues = json.load(f)
            for dialogue in dialogues:
                processed_data[dialogue['dialogue_id']] = {}
                for idx,turn in enumerate(dialogue['turns']):
                    if turn['speaker'] == "SYSTEM":
                        assert output
                        output_data = output.pop(0)
                        processed_data[dialogue['dialogue_id']][turn['turn_id']] = {
                            'start':output_data['beginning'],
                            'end':output_data['end'],
                            'mod':output_data['response']
                        }
        assert not output
        json.dump(processed_data,open(args.output_path,"w"),indent=2)
    else:
        # raise NotImplementedError
        i = 0
        failure = []
        dialogues_list = list(args.test_dir.glob('*.json'))
        dialogues_list.sort()
        processed_data = {}
        print("*****-----*****----- post processing state -----*****-----*****")
        for dialogues_path in dialogues_list:
            with open(dialogues_path) as f:
                dialogues = json.load(f)
            for dialogue in dialogues:
                processed_data[dialogue['dialogue_id']] = {}
                for idx,turn in enumerate(dialogue['turns']):
                    if turn['speaker'] == "SYSTEM":
                        assert output
                        output_data = output.pop(0)
                        for state in output_data['belief']:
                            state = state.split(" ")
                            if len(state) < 3 or state[0]==state[1]:
                                if i not in failure:
                                    failure.append(i)
                                continue
                            processed_data[dialogue['dialogue_id']][f'{state[0]}-{state[1]}'] = " ".join(state[2:])
                            i += 1
        json.dump(processed_data,open(args.output_path,"w"),indent=2)
        write_csv(processed_data,str(args.output_path).replace('json','csv'))
        if failure:
            pickle.dump(failure,open('failure_index.pkl','wb'))

    return



if __name__ == "__main__":
    args = parse_args()
    main(args)