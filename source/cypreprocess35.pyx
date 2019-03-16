import re
from urllib.parse import unquote

# from numpy
from numpy import int as np_int
from numpy import zeros

# from Cnumpy
from numpy cimport ndarray
from numpy cimport int_t as cnp_int

import pickle

def print_list():
	return cprint_list()

def get_detail(images, attacklist, pred, media):
	return cget_detail(images, attacklist, pred, media)

def replace_word2unknown(images, word2idx, maxword):
	return creplace_word2unknown(images, word2idx, maxword)

def preprocess_text(rwords, maxlen):
	return cpreprocess_text(rwords, maxlen)

def find_multiwords_for_media(rword, text, highlight_idx, maxword, n_range):
	return cfind_multiwords_for_media(rword, text, highlight_idx, maxword, n_range)

def find_multiwords_for_purpose(rword, text, start, end, maxword, func, purpose_word):
	return cfind_multiwords_for_purpose(rword, text, start, end, maxword, func, purpose_word)

cdef public object get_reallen(ndarray[cnp_int, ndim=2] images, word2index):
	padidx = word2index['pad']
	reallen_list = [(img.tolist()+[padidx]).index(padidx) for img in images]
	return reallen_list

cdef public object get_zeropad(imagelen, n_dimen):
	return zeros([imagelen, 1, n_dimen, 1])

cdef public object read_picklefile( filename ) :
	word2index = pickle.load(open(filename, 'rb'))
	return word2index

cdef public int calc_anyall(idbool, setfunc):
	if 'all' == setfunc:
		return int(all(idbool))

	# default is 'any'
	return int(any(idbool))

cdef public object cprint_list():
	tmp = [['aaa', 'bbb'], ['ccc', 'ddd']]
	return tmp

cdef public object cget_detail(ndarray[cnp_int, ndim=2] images, attacklist, pred, media):
	# load attacklist
	cdef object img
	cdef int imageslen = len(images)
	cdef ndarray[object, ndim=1] pred_detail = zeros((imageslen,),dtype=(object))
	pred_detail[:] = media

	for i in xrange(imageslen):
		#start_i, end_i = highlight_idx[i]
		tmp_attackstr = []
		img = images[i]
		#img = images[i][start_i:end_i]
		for attack in attacklist:
			attkey, setfunc = attack.split(',')
			idbool = [att in img for att in attacklist[attack]]
			idbool = calc_anyall(idbool, setfunc)

			# string pattern matched
			if True == idbool:
				tmp_attackstr.append(attkey)

		if len(tmp_attackstr) > 0:
			pred_detail[i] = '`'.join(tmp_attackstr)

	return pred_detail.tolist()

cdef public object cpreprocess_text(rwords, maxlen):
	cdef object rwdtmp
	cdef int rwordslen = len(rwords)
	cdef ndarray[object, ndim=2] images = zeros((rwordslen,),dtype=(object,maxlen))

	dotnum = re.compile(r'(\d+\.)[\d+\.]+')
	number = re.compile(r'(\s(\-|)\d+)')
	opscpt = re.compile(r'((\%3c)|<)script((\%3e)|>|\s)', flags=re.IGNORECASE)
	clscpt = re.compile(r'((</)|(\%3c\%2f))script[\s|\+]?(>|(\%3e))', flags=re.IGNORECASE)

	# added from 20170907.preprocess_word
	ssspst = re.compile(r'\$\_post\[.*\]', flags=re.IGNORECASE)
	opeval = re.compile(r'(\@|\<\%)eval', flags=re.IGNORECASE)
	cleval = re.compile(r'\%\>', flags=re.IGNORECASE)
	opphps = re.compile(r'\<\?php', flags=re.IGNORECASE)
	giftag = re.compile(r'gif89a(g)?', flags=re.IGNORECASE)
	for i in xrange(rwordslen):
		rwdtmp = rwords[i]
		rwdtmp = unquote(rwdtmp.lower())

		rwdtmp = re.sub(dotnum, ' dot_num ', rwdtmp)
		rwdtmp = re.sub(opscpt, ' [opscript] ', rwdtmp)
		rwdtmp = re.sub(clscpt, ' [clscript] ', rwdtmp)
		rwdtmp = re.sub(ssspst, ' $_post ', rwdtmp)
		rwdtmp = re.sub(opeval, ' [opeval] ', rwdtmp)
		rwdtmp = re.sub(cleval, ' [cleval] ', rwdtmp)
		rwdtmp = re.sub(opphps, ' [opphps] ', rwdtmp)
		rwdtmp = re.sub(giftag, ' [giftag] ', rwdtmp)
		for ch in ['\r\n',';','(',')','+','=','/',',',':','{','}','\'','"','.', '`']:
			if ch in rwdtmp:
				rwdtmp = rwdtmp.replace(ch, '  ')
		rwdtmp = re.sub(number, ' number ', rwdtmp)

		rwdtmp = rwdtmp.split()
		rwdtmplen = len(rwdtmp)

		if len(rwdtmp) > maxlen:
			images[i] = rwdtmp[:maxlen]
			rwords[i] = rwdtmp
			continue
		images[i] = rwdtmp + ['pad']*(maxlen - len(rwdtmp))
		rwords[i] = rwdtmp + ['pad']*(maxlen - len(rwdtmp))
	return images

cdef public ndarray[cnp_int, ndim=2] creplace_word2unknown(images, word2idx, maxword):
	cdef int imageslen = len(images)
	cdef int imagesize = len(images[0])
	cdef ndarray[cnp_int, ndim=1] tmp
	cdef object imgtmp
	cdef ndarray[cnp_int, ndim=2] ret_images = zeros((imageslen,),dtype=(np_int,maxword))

	for i in xrange(imageslen):
		tmp = zeros(imagesize, dtype=np_int)
		imgtmp = images[i]
		for j in xrange(imagesize):
			try:
				tmp[j] = word2idx[imgtmp[j]]
			except KeyError:
				tmp[j] = word2idx['unknown_token']
		ret_images[i] = tmp

	return ret_images

cdef public object get_word2index(filename):
	word2index = pickle.load(open(filename, 'r'))
	return word2index

cdef public object cfind_multiwords_for_media(rword, text, highlight_idx, maxword, n_range):
	start, end = [highlight_idx - 10, highlight_idx + 10]
	start, end = max(0, start), min(maxword, end)
	cdef object word

	for ch in ['\r\n', '\n', '\r']:
		text = text.replace(ch, len(ch)*' ')

	# regex definition
	dotnum_str = '(\d+\.)[\d+\.]+'
	number_str = '\d+'
	opscpt_str = 'script'
	clscpt_str = 'script'
	ssspst_str = 'post'
	opeval_str = 'eval'
	opphps_str = 'php'
	giftag_str = 'gif89'

	alpha_numeric = re.compile('^[A-Za-z0-9]{2,}$')

	# generate regex
	generated_regex_str = []

	# to match with regex, these should be escaped
	special_regex = ['\\', '^', '$', '.', '*', '+', '?', '|', '{', '}', '[', ']', '(', ')']

	# among must-escaped characters, these should be replaced to SPACE
	ignore_regex = ['|']
	for i in range(start, end):
		try:
			word = rword[i]
			if 'number' == word:
				continue
				#generated_regex_str.append(number_str)
			elif 'dot_num' == word:
				continue
				#generated_regex_str.append(dotnum_str)
			elif '[opscript]' == word:
				generated_regex_str.append(opscpt_str)
			elif '[clscript]' == word:
				generated_regex_str.append(clscpt_str)
			elif '$_post' == word:
				generated_regex_str.append(ssspst_str)
			elif '[opeval]' == word:
				generated_regex_str.append(opeval_str)
			elif '[cleval]' == word:
				# it should continue, because cleval_str does not contain any single alpha-numeric character
				continue
			elif '[opphps]' == word:
				generated_regex_str.append(opphps_str)
			elif '[giftag]' == word:
				generated_regex_str.append(giftag_str)
			else:
				word_check = alpha_numeric.match(word)
				if None == word_check:
					continue
				else:
					generated_regex_str.append(word)

		except IndexError:
			break

	# search patterns on the original text
	generated_regex_str = '(.+?)'.join(generated_regex_str)
	generated_regex = re.compile(generated_regex_str, flags=re.IGNORECASE)
	result = generated_regex.search(text)

	if None == result:
		return -1, -1, generated_regex_str

	start_i, end_i = list(result.span(0))
	return [start_i, end_i, generated_regex_str]

cdef public object cfind_multiwords_for_purpose(rword, text, offset, offset_str, maxword, func, purpose_word):
	cdef object word
	if 'any' == func:
		# find word that matches to text string
		for pw in purpose_word:
			if pw in rword:
				word = pw
				regex = re.compile(word, flags=re.IGNORECASE)
				break

		# exception handling for 'pad'
		try:
			result = regex.search(text)
			if None == result:
				return -1, -1
			result = regex.search(text)
			start_i, end_i = list(map(lambda x: x+offset_str, list(result.span(0))))
		except UnboundLocalError:
			start_i, end_i = -1, -1

		return start_i, end_i

	elif 'all' == func:
		for pw in purpose_word:
			word = pw
			regex = re.compile(word, flags=re.IGNORECASE)
			result = regex.search(text)
			if None == result:
				return -1, -1
			start_i, end_i = list(map(lambda x: x+offset_str, list(result.span(0))))
		return start_i, end_i

	else:
		return -1, -1


