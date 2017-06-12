import re

quote_filename = "author-quote.txt"
raw_lines = open(quote_filename).readlines()
quotes = [line.split("\t")[1] for line in raw_lines]
raw = "".join(quotes)
raw = raw.lower()
raw = re.sub('([!\'"#$%&()*+,-./:;=?£—_’])', r' \1 ', raw)
raw = re.sub(' {2,}', ' ', raw)
out = open("quote.txt", 'w')
out.write(raw)
