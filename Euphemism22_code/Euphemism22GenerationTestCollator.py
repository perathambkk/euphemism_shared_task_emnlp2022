#!/usr/bin/env python

"""
Author: Peratham Wiriyathammabhum


"""
import torch

class Euphemism22GenerationTestCollator(object):
	"""
	Data Collator used in a classification rask. 
	
	It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
	can go straight into a transformer model.

	This class is built with reusability in mind: it can be used as is as long
	as the `dataloader` outputs a batch in dictionary format that can be passed 
	straight into the model - `model(**batch)`.

	Arguments:

	  use_tokenizer (:obj:`transformers.tokenization_?`):
		  Transformer type tokenizer used to process raw text into numbers.

	  labels_ids (:obj:`dict`):
		  Dictionary to encode any labels names into numbers. Keys map to 
		  labels names and Values map to number associated to those labels.

	  max_sequence_len (:obj:`int`, `optional`)
		  Value to indicate the maximum desired sequence to truncate or pad text
		  sequences. If no value is passed it will used maximum sequence size
		  supported by the tokenizer and model.

	"""
	def __init__(self, use_tokenizer, max_sequence_len=None):
		# Tokenizer to be used inside the class.
		self.use_tokenizer = use_tokenizer
		# Check max sequence length.
		self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

		return

	def __call__(self, sequences):
		"""
		This function allowes the class objesct to be used as a function call.
		Since the PyTorch DataLoader needs a collator function, I can use this 
		class as a function.

		Arguments:

		  item (:obj:`list`):
			  List of texts and labels.

		Returns:
		  :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
		  It holds the statement `model(**Returned Dictionary)`.
		"""
		# Get all texts from sequences list.
		texts = [sequence['text'] for sequence in sequences]
		# Get all labels from sequences list.
		labels = [sequence['id'] for sequence in sequences]
		targettexts = [str(sequence['id']) for sequence in sequences]
		# Call tokenizer on all texts to convert into tensors of numbers with 
		# appropriate padding.
		inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
		targets = self.use_tokenizer(text=targettexts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
		# Update the inputs with the associated encoded labels as tensor.
		allinputs = {'source_ids':inputs['input_ids'], 'source_mask':inputs['attention_mask'], 
					'target_ids':targets['input_ids'], 'target_mask':targets['attention_mask'],
					'labels':torch.tensor(labels)}

		return allinputs

