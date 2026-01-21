"""
This script is adapted from
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import tiktoken
import os
import pdb
import glob
import jieba
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import argparse
from rouge_score import rouge_scorer
from tqdm import tqdm
import sys
import os
from datetime import datetime, timezone
import time
import torch
from torch.utils.data import DataLoader, Dataset


from data.PaulGrahamEssays.loader import Needle
from models.loader import load_model
from utils import *
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="data/PaulGrahamEssays", # PaulGrahamEssays
                 retrieval_question="The best thing to do in San Francisco is: ",
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 40,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model='',
                 model_name_suffix=None,
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True,
                 step=100,
                 method='pyramidkv',
                 attn_implementation='flash_attention_2',
                 max_capacity_prompt=128):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.method = method
        self.max_capacity_prompts = max_capacity_prompt
        self.attn_implementation = attn_implementation

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
                self.context_lengths = np.array([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8000])
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

        self.model = model

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args, model, tokenizer, dataloader):
        for batch in tqdm(dataloader):
            data = {}
            doc_ids, prompt_ids, params, depth, context_length= batch[0]['doc_ids'], batch[0]['prompt_ids'], batch[0]['params'], batch[0]['depth'], batch[0]['context_length']
            self.bound_evaluate_and_log(args, context_length, depth, prompt_ids, params)

    def evaluate_and_log(self, args, context_length, depth_percent, prompt_ids, params):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(args, context_length, depth_percent):
                print("result exists, skipping")
                return
            else:
                print("result does not exist, testing")

        input_ids = torch.tensor([prompt_ids]).to('cuda')
        past_key_values = None
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to('cuda')
        input = {'input_ids': input_ids, 'past_key_values': past_key_values, 'position_ids': position_ids} # pos 这里只支持bs 1
        stop_list = get_stop_tokens(args, tokenizer)

        # generate
        if args.reuse == 'no' and args.drop == 'False': # 全部重算+全kv
            response, ttft, tpot = vanilla(args, model, tokenizer, input, stop_list, max_new_tokens, {})
        
        else:
            extra_config = initialize_config(args)
            extra_config['other_config']['data_params'] = params

            if args.reuse != 'no':
                extra_config['reuse_config']['cat_kv'] = load_kv(args, None, None, None, None, f'{str(context_length)}_{str(depth_percent)}')

            response, ttft, tpot = decode(args, model, tokenizer, input, stop_list, max_new_tokens, extra_config)

        print(response)

        if len(response) != 0:
            if 'eat a sandwich and sit in Dolores Park on a sunny day' in response:
                score = 1
            else:
                score = scorer.score(self.needle, response)['rouge1'].fmeasure
        else:
            score = 0.0

        results = {
            'model' : self.model,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'),
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'{args.output_path}'):
                os.makedirs(f'{args.output_path}')

            # Save the result to file for retesting
            p = f'{args.output_path}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f, ensure_ascii=False)

    def result_exists(self, args, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = f'{args.output_path}/'
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args, model, tokenizer, dataloader):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args, model, tokenizer, dataloader)

def get_stop_tokens(args, tokenizer):
    lst = [tokenizer.bos_token_id]

    lst.append(128009)
    lst.append(128006)

    lst.append(tokenizer.encode('\n', add_special_tokens=False)[-1])
    
    return lst

if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', default=512, metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', default=8000, metavar='N', type=int, help='a number')
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "None"])
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA3", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=512)

    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--reuse', type=str, default='fp16')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--kv_path', type=str, default=None)
    parser.add_argument('--rate', type=float, default=0.15)
    parser.add_argument('--drop', type=str, default=False)
    parser.add_argument('--drop_config', type=str, default=None)
    args = parser.parse_args()

    ht = LLMNeedleHaystackTester(model=args.model,
                                 model_name_suffix=None,
                                 model_provider=args.model_provider,
                                 context_lengths_min=args.s_len,
                                 context_lengths_max=args.e_len,
                                 step=args.step,
                                 )

    # load_datasets
    print(f'loading {args.dataset}...')
    def custom_collate_fn(batch):
        return batch
    
    dataset = Needle(args)
    dataloader = DataLoader(dataset, collate_fn=custom_collate_fn)
    max_new_tokens = 64

    # load_model
    print(f'loading {args.model}')
    model, tokenizer = load_model(args)

    ht.start_test(args, model, tokenizer, dataloader)
