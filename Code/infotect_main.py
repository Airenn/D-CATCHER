from scapy.all import* # packet 관련 라이브러리
import threading # thread 관련 라이브러리
from multiprocessing import Process, Queue, Manager
import time # 시간 값 관련 라이브러리
import datetime #시간 관련 라이브러리
import dbcon # DB 연결 및 조회 등록 클래스 저장된 .py
import smtplib
import json
import math
import pickle
import joblib
import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr

from_user = 'onse3714@nate.com'
to_user = 'ksj.infotect@gmail.com'
to_user_name = 'ksj.infotect'
mail_subject = 'DGA Detected'

class bcolors: #Print 색상 지정
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def showPacket(packet):
	data = []
	tmp = []
	src_ip = [] #src ip를 임시로 저장할 리스트

	global rr_data
	global qr_queue
	global src_dic #src ip를 관리하는 dic

	if packet.getlayer(DNS).an == None and packet.getlayer(DNS).qd.qtype != 28: # dns packet sniffing AAAA 레코드 제외
		tmp = repr(packet[DNSQR])
		data = tmp[15:-23] # DNS QR 내용 저장 및 Domain 추출
		qr_queue.put(data)

		src_ip = repr(packet[IP].src)
		src_dic[data] = src_ip[1:-1]

		print(f"QR Domain :  {data}")

	elif packet.getlayer(DNS).an != None and packet.getlayer(DNS).qd.qtype != 28: # dns packet RR sniffing AAAA 레코드 제외
		if packet.getlayer(DNS).an.rdata != None:
			count = packet[DNS].ancount # DNS RR 깊이 체크
			r_name = ""
			for x in range(packet[DNS].ancount): # DNS RR Domain 추출
				tmp = repr(packet[DNSRR][x])
				count = tmp.count("=")
				sp = tmp.split("=")
				if r_name == "":
					r_name = sp[1][1:-7]
				if sp[6][0] == "'":
					data = sp[6][1:-18]
				else:
					if count < 7 :
						data = sp[6][0:-3]
					else:
						data = sp[6][0:-15]
			rr_data[r_name] = data # dic {QR : IP}의 형태로 저장
			src_ip = repr(packet[IP].dst)
			src_dic[r_name] = src_ip[1:-1]
 
#			print("--------------------\n")
#			for x in rr_data.items():
#				print(x)
			print("--------------------\n")
	
def sniffing():
	sniff(filter="udp port 53",prn=showPacket,count=0)


def search_qr(qr_queue, testee_queue):
	mysql_controller = dbcon.MysqlController()

	global rr_data
	global src_dic
	global testee_list

	while(1):
		print("search_qr list : ",qr_queue.qsize())  #변경 되는 Queue 값 체크
		try:
			domain = qr_queue.get() # Queue에서 하나 빼서 저장
			sp_domain = domain.split('.')
			sp_domain.pop(0)
			sp_domain = '.'.join(sp_domain)
			if mysql_controller.db_search_cal(sp_domain) != None: 
				time.sleep(0.3)
				print("  #  "+domain+"  #  ")
				print(bcolors.HEADER + "\n  Allow List Check  \n"+bcolors.ENDC)
				if rr_data.get(domain):
					del rr_data[domain]
				if src_dic.get(domain):
					del src_dic[domain]
			elif mysql_controller.db_search_today(domain) != None:
				time.sleep(0.3)
				print("  #  "+domain+"  #  ")
				print(bcolors.HEADER+"\n Today List Check \n"+bcolors.ENDC)
	#			print("get result : ",rr_data.get(domain))
				if rr_data.get(domain):
					del rr_data[domain]
				if src_dic.get(domain):
					del src_dic[domain]
			elif mysql_controller.db_search_cdl(domain) != None:
				time.sleep(0.3)
				print("  #  "+domain+"  #  ")
				print(bcolors.HEADER+"\n Deny List Check \n"+bcolors.ENDC)
				if rr_data.get(domain):
					del rr_data[domain]
				if src_dic.get(domain):
					del src_dic[domain]
			elif mysql_controller.db_search_ctl(domain) != None:
				time.sleep(0.3)
				print("  #  "+domain+"  #  ")
				print(bcolors.FAIL+"\n Detect List Check \n"+bcolors.ENDC)
				if rr_data.get(domain):
					del rr_data[domain]
				if src_dic.get(domain):
					del src_dic[domain]
			else: # 질의 후 값이 없는 경우
				testee_queue.put(domain)



		except IndexError: # Queue 값이 없을 경우 예외 처리
#			print()
#			print("---------------------------------------------\n")
			print(" Serch_qr() IndexError! ", qr_queue.Empty())
#			print("---------------------------------------------\n")
#			print()
#			break


################################### 피처 함수 ########################################

def Entropy(df): # 엔트로피 함수
        entropy = []
        counts = [Counter(i) for i in list(df["Domain"])]
        for domain in counts:
            prob = [ float(domain[c]) / sum(domain.values()) for c in domain ]
            entropy.append(-(sum([ p * math.log(p) / math.log(2.0) for p in prob ])))
        return entropy


def ML_start():

############################## 전처리에 필요한 txt #################################

	with open("googlebooks-eng-10000.txt", "r") as f:
        	word_list = [line.rstrip() for line in f if len(line) > 3 ]
	f.close()
	with open("TLD_list.txt", "r") as g:
        	TLD_list = [line.rstrip() for line in g]
	g.close()

	file=open("3gram","rb")
	three_gram=pickle.load(file)

	file=open("4gram","rb")
	four_gram=pickle.load(file)

	file=open("5gram","rb")
	five_gram=pickle.load(file)

	std_scaler = pickle.load(open('StandardScaler.pkl','rb'))

################################저장된 머신러닝 불러옴##################################

	clf_from_joblib = joblib.load('RandomForestClassifier_2000000.pkl')

#################################### 새로운 데이터 입력 ####################################

	string_list =  pd.DataFrame({"Domain":[]})
	Test_df = pd.DataFrame()


	global testee_queue
	global qr_queue
	global src_dic

	while(1):
#		print("testee_list : ",testee_queue.qsize())  #변경 되는 Queue 값 체크
#		print("qr_queue_list : ",qr_queue.qsize())  #변경 되는 Queue 값 체크
		try:
			domain = testee_queue.get()
			Test_df = string_list.append( {'Domain' : domain}, ignore_index=True) #Queue에서 하나 빼서 저장
#			print(string_list)

 ##################################### 새로운 데이터 전처리 ###############################################

			Test_df["TLD"] = list(itertools.chain.from_iterable([[next((tld for i, tld in enumerate(TLD_list) if Domain[-len(tld):] == tld), Domain[Domain.rfind("."):])] for Domain in Test_df.Domain])) # 도메인 TLD
			Test_df["Sub_Domain"] = [Test_df.Domain.str[:-len(Test_df.TLD.str)] for Test_df.Domain.str, Test_df.TLD.str in zip(Test_df.Domain, Test_df.TLD)]
			Test_df["TLD_index"] = list(itertools.chain.from_iterable([[next(((i+1) for i, tld in enumerate(TLD_list) if Domain[-len(tld):] == tld), 0)] for Domain in Test_df.Domain]))
			Test_df["3-gram_Score"] = [(sum([three_gram.get(Domain[i:i+3]) for i in range(len(Domain)-2) if(Domain[i:i+3] in three_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
			Test_df["4-gram_Score"] = [(sum([four_gram.get(Domain[i:i+4]) for i in range(len(Domain)-2) if(Domain[i:i+4] in four_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
			Test_df["5-gram_Score"] = [(sum([five_gram.get(Domain[i:i+5]) for i in range(len(Domain)-2) if(Domain[i:i+5] in five_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
			Test_df["Length"] = Test_df.Sub_Domain.str.len()# 도메인 길이
			Test_df["Numeric_ratio"] = Test_df.Sub_Domain.str.count('[0-9]') / Test_df.Sub_Domain.str.len()# 도메인에 포함된 숫자 개수 / 도메인 길이
			Test_df["Vowel_ratio"] = Test_df.Sub_Domain.str.count('[a,e,i,o,u]') / Test_df.Sub_Domain.str.len()# 도메인에 포함된 모음 개수 / 도메인 길이
			Test_df["Consonant_ratio"] = Test_df.Sub_Domain.str.count('[b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z]') / Test_df.Sub_Domain.str.len()#도메인에 포함된 자음 개수 / 도메인 길이
			Test_df["Consecutive_consonant"] = Test_df.Sub_Domain.str.count('[^.aeiou]{3,}') # 연속되는 3글자(자음) 개수
			Test_df["Consecutive_Vowel"] = Test_df.Sub_Domain.str.count('[aeiou]{2,}')# 연속되는 2글자(모음) 개수
			Test_df["period"] = Test_df.Sub_Domain.str.count('[.]')#개수
			Test_df["Entropy"] = Entropy(Test_df)# Shannon 엔트로피
			Test_df["Max_Consecutive_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Test_df.Sub_Domain.str.findall('[^.aeiou]{3,}')] # 연속되는 3글자(자음) 최대 값
			Test_df["Max_voewl_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Test_df.Sub_Domain.str.findall('[aeiou]{2,}')] # 연속되는 3글자(모음) 최대 값
			Test_df["Meaning_count"] = [len([word for word in word_list if(word in Domain) ]) for Domain in Test_df["Sub_Domain"].to_list()]

 ####################################################### 실시간 데이터 예측 #######################################################

			Test_data = Test_df.iloc[:, 3:]
			test_scaler = std_scaler.transform(Test_data)
			Test_data = pd.DataFrame(test_scaler, columns=Test_data.columns, index=list(Test_data.index.values))
			Test_Predict = clf_from_joblib.predict(Test_data) # 검증용 데이터셋 정답 예측
			print(domain, Test_Predict)
			string_list = string_list[0:0]
			Test_df = pd.DataFrame()

			mysql_controller = dbcon.MysqlController()

			if mysql_controller.db_search_today(domain) != None: #DB에 질의 후 없을 경우 Domain 추가
				print("  #  "+domain+"  #  ")
				print(bcolors.HEADER+"\nToday List Check\n"+bcolors.ENDC)
			else:

				tmp = str(Test_Predict)
				tmp = tmp[2:-2]
				mysql_controller.db_insert_today(domain,src_dic[domain],tmp) # Domain과 dic에 IP를 저장


				if Test_Predict == 'DGA':
					mysql_controller.db_insert_ctl(domain,src_dic[domain],rr_data[domain]) # Domain과 dic에 IP를 저장
					f = open("./domain.txt", 'w')
					f.write(domain)
					f.close()
					os.system('./mailer.py')
					
				if src_dic.get(domain):
#					print(" src_dic[ip] 삭제 : ", src_dic[domain])
					del src_dic[domain]

				if rr_data.get(domain):
#					print("rr_data[domain] 삭제 : ", rr_data[domain])
					del rr_data[domain]


		except KeyError: # rr_data 값이 없을 경우 예외 처리(NX domain)
			print()
			print("---------------------------------------------\n")
			print("                NX Domain Check                ")
			print("---------------------------------------------\n")
			print()
			continue


if __name__ =='__main__':
	qr_queue = Queue()
	manager = Manager()
	rr_data = manager.dict()
	src_dic = manager.dict()
	testee_queue = Queue()

	print(bcolors.FAIL+"  _____   "+bcolors.ENDC+"           _____      _       _ ")              
	print(bcolors.FAIL+" |  __ \  "+bcolors.ENDC+"          / ____|    | |     | |")              
	print(bcolors.FAIL+" | |  | | "+bcolors.ENDC+" ______  | |     __ _| |_ ___| |__   ___ _ __ ")
	print(bcolors.FAIL+" | |  | | "+bcolors.ENDC+"|______| | |    / _` | __/ __| '_ \ / _ \ '__|")
	print(bcolors.FAIL+" | |__| | "+bcolors.ENDC+"         | |___| (_| | || (__| | | |  __/ |   ")
	print(bcolors.FAIL+" |_____/  "+bcolors.ENDC+"          \_____\__,_|\__\___|_| |_|\___|_|   ")
	print("   _____ _______       _____ _______   _   _            ")
	print("  / ____|__   __|/\   |  __ \__   __| | | | |           ")
	print(" | (___    | |  /  \  | |__) | | |    | | | |           ")
	print("  \___ \   | | / /\ \ |  _  /  | |    | | | |           ")
	print("  ____) |  | |/ ____ \| | \ \  | |    |_| |_|           ")
	print(" |_____/   |_/_/    \_\_|  \_\ |_|    (_) (_)           ")
	print()                                                 
	print()                                             

	t1 = threading.Thread(target=sniffing)
	t1.start()

	time.sleep(10)
	p2 = Process(target=search_qr, args=(qr_queue,testee_queue)).start()
	p3 = Process(target=ML_start).start()


