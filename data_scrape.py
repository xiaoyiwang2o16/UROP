import glob 
import csv
def load_data(filter_str):
	# filter_str is a string of the form '<country>.<gender>.<child>'
	percentages = [filter_str]
	for field in ['openface', 'openpose', 'physiology']:
		files = glob.glob(filter_str+'*'+field+'*')
		total =0 
		numLines = 0
		for file in files:
			with open(file) as csvfile:
				reader = csv.reader(csvfile, delimiter=" ", quotechar= '|')
				
				for line in reader:
					numLines += 1
					newLine = line[0].split(",")
					newLine = list(map(float, newLine))
					if sum(newLine[7:-3]) == 0:
						total += 1
		percentages.append(total)
	return percentages

#load_data('1.0.20', 'physiology')
def get_all_data():
	people = []
	fields = []
	for i in [0, 1]:
		for j in [0, 1]:
			if i == 0:
				for k in range(1, 18):
					people.append(str(i)+'.'+str(j)+'.'+str(k))
			else:
				for k in [1, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
					people.append(str(i)+'.'+str(j)+'.'+str(k))	

	data = []
	
	#for person in people:
	for person in ['0.0.1', '0.0.2']:
		data.append(load_data(person))
	print(data)
	with open('statistics.csv', 'w') as csvfile:
		#filewriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
		filewriter = csv.writer(csvfile, delimiter=',')

		for row in data:
			row = list(map(str, row))
			filewriter.writerow(row)


get_all_data()

