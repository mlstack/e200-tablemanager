<h1>Training Requirement</h1>

<h2>교육 범위</h2>
<ul>
<li>1안 - Big Data 경험</li>
  <ul>
    <li>Big Data Architecture</li>
    <li>Hadoop Eco-System</li>
    <li>Hadoop Data Processing</li>
  </ul>
<li>2안 - Analytics with Small Data</li>
  <ul>
    <li>Statistic</li>
    <li>Machine Learning</li>
    <li>Deep Learning</li>
  </ul>
<li>3안 - Analytics with Big Data</li>
  <ul>
    <li>Maching Learning using Hadoop/Spark</li>
    <li>Deep Learnning using Spark</li>
   </ul>
</ul>

<h2>교육 훈련의 방향</h2>
<ul>
<li>1안 - 교양 강좌식 - "저런게 있구나..."</li>
<li>2안 - 넓게 - 숲 보기</li>
<li>3안 - 깊게 - 숲속의 나무 보기</li>
<li>4안 - 넓고 깊게 - 우리 숲은 이렇게 생겼어...</li>
</ul>





<h2>교육 방식</h2>
<ul>
<li>1안 - Semi Project 수행</li>
<li>2안 - 실전 Project 수행</li>
<li>3안 - Step By Step Training</li>
</ul>

<ul>
  <li></li>
</ul>

<pre>
테이블 매니져 ㅇㅇㅇ 교육
강사 : 이이백
Phone : 010-4527-5033
eMail : Yibeck.Lee@gmail.com
교육생 : 이상엽, 임창용, 김동규, 김준성

Topic : 레스토랑별 고객수 예측
	- 1주의 월,화,수,목,금,토,일
	- 시간대
	- 팀수, 총인원
		- 다인수
	- 소주제
		- NoShow 가능성 Score
		- 취소 가능성 Score
[Comment]
	- 심리 Features 부재하의 예측
	- 시계열 예측이 가장 높은 난이도의 데이터 분석을 요구
	- 외부 데이터 획득 방안
	- Why?(왜 그런데...)의 요구 
		- 차주 예약수 20-25-35-20-25-35   -> 설명할 수 없음.
		- 차선책 = Machine Learning(Decision Tree & Random Forest)
	- Service 의 지속가능성
1안) AWS AI-as-a-Service
- Forecast : timeseries
- 레스토랑별 고객수 예측
	- 예측 수 만큼 레스토랑에서 준비
	- 특징 : 날씨, 위치, View, ....
	- Analytics 결과의 활용
		- Discount Ticket
		- Overbooking 
- 구현 - AWS에서 제공하는 분석서비스

2안) 자체적 수요예측 System 구축
	- System Architecture
	- Analytics Component
	- Analytics - Deep Learning 지식 획득
	- Deep Learning의 프로그램 구현


Entity & Attributes
- Order, Payment, 

System
- AWS Cloud : EC2, --> Lambda
- DBMS : mongodb, mysql
- Web 
	- Server : Node.js
	- Client : Angular
- Dev Tools
	- Visual Code, Sublime, 
	- MackBook Console, Putty

- AI Dev System 
	- Server : AWS Lambda or AI Local Server
		- OS : Ubuntu CentOS
		- JDK
		- Node/NPM
		- Spark : Memory based Bigdata Technology <- Hadoop, Local
			- Json to Spark/SQL - org.apache.spark.sql
			- ETL : Extract Transform Loading
			- Streaming (based 예측 - AI Service ) - org.apache.spark.streaming
			- Machine Learning Package - Spark/ML  - org.apache.spark.mllib
			- Deep Learning의 Native Component 제공(neural-network)
				- Tensorflow OnSpqrk : Yahoo
				- Deeplearning4j : Spark를 이용한 Java/Scala 기반의 AI

			- Native : Scala
				- API : Java, Scala, Python, R 

	- Capacity
		- JavaScript, R, Python, Java 


[Demo]
- 동영상
- Node.Js/NPM Install 



[Work & Study]
- Python
	- Module
		- Numpy, Pandas, Scikit-Learn Tensorflow 
	- Windows : Anaconda3  - (Numpy, Pandas, Scikit-Learn,) Tensorflow
	- Linux : python.org - download
- Hadoop/hdfs Command
	- hdfs -ls 
- Spark Install

</pre>



