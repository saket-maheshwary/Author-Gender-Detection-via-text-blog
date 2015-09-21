#importing natural language processing toolkit library
import nltk
from nltk.stem.porter import PorterStemmer
from itertools import groupby
import math
from sklearn.metrics import confusion_matrix, classification_report
import numpy
from nltk.util import bigrams

#importing decimal for precision
from decimal import *
getcontext().prec = 4

#importing svm using scikit learn libray
from sklearn import svm
import sys

#sample list
samplelist=[]
#male-0, female-1
classes=[]

#normalization
#min list
minlist=[]

#max list
maxlist=[]

flag=False
def parse(fileName):
   print fileName
   global flag
   f = open(fileName)
   xml = f.read()
   xml = xml.replace('\n', '').replace('\r', '')
   post_list = xml.split('<date>')
   blog_list = []
   for post in post_list:
     try:
       date = post.split('</date>')[0]
       post = post.split('<post>')[1].split('</post>')[0].strip()
       blog_list.append({'post': post})
     except:
       pass
      
   #print blog_list   # return {'blogs': blog_list}
   
   #character based features
   no_of_characters=0
   no_of_letters=0
   no_of_upper_characters=0
   no_of_digits=0
   no_of_white_spaces=0
   no_of_tabspace_characters=0
   no_of_special_characters=0
   no_of_all_characters=0
   
   #word based features
   no_of_words=0
   no_of_unique_words=0
   no_of_words_length_morethan_six=0
   no_of_words_length_lessthan_three=0
   avg_length_word=0
   
   #syntactic features
   no_of_single_quotes=0
   no_of_commas=0
   no_of_periods=0
   no_of_colons=0
   no_of_semi_colons=0
   no_of_question_marks=0
   no_of_multiple_questions_marks=0
   no_of_exclamation_marks=0
   no_of_multiple_exclamation=0
   no_of_ellipsis=0
   total_syntactic_features=0
   
   #function words features
   no_of_article_words=0
   no_of_prosentence_words=0
   no_of_Adposition_words=0
   no_of_conjunction_words=0
   no_of_proposition_words=0
   no_of_auxilary_verbs=0
   no_of_interjection_words=0

   for i in range(len(blog_list)):
      no_of_characters+=blog_list[i]['post'].count('')
      no_of_digits+=sum(c.isdigit() for c in blog_list[i]['post'])
      no_of_letters+=sum(c.islower() for c in blog_list[i]['post'])
      no_of_white_spaces+=sum(c.isspace() for c in blog_list[i]['post'])
      no_of_upper_characters+=sum(c.isupper() for c in blog_list[i]['post'])
      no_of_tabspace_characters+=blog_list[i]['post'].count('\t')
      
      #caluculating syntactic based features
      no_of_single_quotes+=blog_list[i]['post'].count('\'')
      no_of_commas+=blog_list[i]['post'].count(',')
      no_of_periods+=blog_list[i]['post'].count('.')
      no_of_colons+=blog_list[i]['post'].count(':')
      no_of_semi_colons+=blog_list[i]['post'].count(';')
      no_of_question_marks+=blog_list[i]['post'].count('?')
      no_of_multiple_questions_marks+=blog_list[i]['post'].count('???')
      no_of_exclamation_marks+=blog_list[i]['post'].count('!')
      no_of_multiple_exclamation+=blog_list[i]['post'].count('!!!!')
      no_of_ellipsis+=blog_list[i]['post'].count('...')
            
      #text3=blog_list[i]['post'].split()
      blog_list[i]['post']=unicode(repr(blog_list[i]['post']))
      text=nltk.word_tokenize(blog_list[i]['post'])
      #print text.categories()
      no_of_words+=len(text)
      for word in text:
         words.add(word)
         avg_length_word+=len(word)
         if len(word)>6:
            no_of_words_length_morethan_six=no_of_words_length_morethan_six+1
         if len(word)<4:
            no_of_words_length_lessthan_three=no_of_words_length_lessthan_three+1 
      #print text
      postagtext=nltk.pos_tag(text)
      #print postagtext
      no_of_article_words+=str(postagtext).count('DT')
      no_of_Adposition_words+=str(postagtext).count('IN')
      no_of_conjunction_words+=str(postagtext).count('CC')
      no_of_proposition_words+=str(postagtext).count('PRP')
      no_of_prosentence_words+=(blog_list[i]['post'].count('yes')+blog_list[i]['post'].count('no')+blog_list[i]['post'].count('okay')+blog_list[i]['post'].count('OK'))
      no_of_auxilary_verbs+=str(postagtext).count('VBP')+str(postagtext).count('VBD')
      no_of_interjection_words+=str(postagtext).count('UH')
      #bigramtext=nltk.BigramTagger(nltk.bigrams(text))
      #print bigramtext
      #for i in range(len(uniquewords)):
      #   set(uniqueword[i])
   #no_of_unique_words=len(set(uniquewords))
   
   no_of_all_characters=(no_of_digits+no_of_letters+no_of_white_spaces+no_of_tabspace_characters)
   total_syntactic_features=(no_of_single_quotes+no_of_commas+no_of_periods+no_of_colons+no_of_semi_colons+no_of_question_marks+no_of_exclamation_marks)
   
   no_of_letters=round(no_of_letters/float(no_of_characters),6)
   no_of_digits=round(no_of_digits/float(no_of_characters),6)
   no_of_white_spaces=round(no_of_white_spaces/float(no_of_characters),6)
   no_of_upper_characters=round(no_of_upper_characters/float(no_of_characters),6)
   no_of_tabspace_characters=round(no_of_tabspace_characters/float(no_of_characters),6)
   no_of_special_characters=no_of_characters-(no_of_all_characters+total_syntactic_features)
   no_of_special_characters=round(no_of_special_characters/float(no_of_characters),6)
   
   no_of_words=float(no_of_words)
   avg_length_word=round(avg_length_word/no_of_words,6)
   no_of_unique_words=round(len(words)/no_of_words,6)
   no_of_words_length_morethan_six=round(no_of_words_length_morethan_six/no_of_words,6)
   no_of_words_length_lessthan_three=round(no_of_words_length_lessthan_three/no_of_words,6)
   #print Hapax_legomena_ratio,Hapax_dislegomena_ratio
   
   no_of_single_quotes=round(no_of_single_quotes/float(no_of_characters),6)
   no_of_commas=round(no_of_commas/float(no_of_characters),6)
   no_of_periods=round(no_of_periods/float(no_of_characters),6)
   no_of_colons=round(no_of_colons/float(no_of_characters),6)
   no_of_semi_colons=round(no_of_semi_colons/float(no_of_characters),6)
   no_of_question_marks=round(no_of_question_marks/float(no_of_characters),6)
   no_of_multiple_questions_marks=round(no_of_multiple_questions_marks/float(no_of_characters),6)
   no_of_exclamation_marks=round(no_of_exclamation_marks/float(no_of_characters),6)
   no_of_multiple_exclamation=round(no_of_multiple_exclamation/float(no_of_characters),6)
   no_of_ellipsis=round(no_of_ellipsis/float(no_of_characters),6)

   no_of_article_words=round(no_of_article_words/no_of_words,6)
   no_of_prosentence_words=round(no_of_prosentence_words/no_of_words,6)
   no_of_Adposition_words=round(no_of_Adposition_words/no_of_words,6)
   no_of_conjunction_words=round(no_of_conjunction_words/no_of_words,6)
   no_of_proposition_words=round(no_of_proposition_words/no_of_words,6)
   no_of_auxilary_verbs=round(no_of_auxilary_verbs/no_of_words,6)
   no_of_interjection_words=round(no_of_interjection_words/no_of_words,6)
   
   list.append(no_of_characters);
   if(flag==False):
      minlist.append(no_of_characters)
   	#minindex=minindex+1;
      maxlist.append(no_of_characters)
   	#maxlist=maxlist+1;
   if(minlist[0]>no_of_characters):
   	minlist[0]=no_of_characters
   if(maxlist[0]<no_of_characters):
   	maxlist[0]=no_of_characters

   list.append(no_of_digits)
   if(flag==False):
   	minlist.append(no_of_digits)
   	#minindex=minindex+1;
   	maxlist.append(no_of_digits)
   	#maxlist=maxlist+1;
   if(minlist[1]>no_of_digits):
   	minlist[1]=no_of_digits
   if(maxlist[1]<no_of_digits):
   	maxlist[1]=no_of_digits

   list.append(no_of_letters)
   if(flag==False):
   	minlist.append(no_of_letters)
   	#minindex=minindex+1;
   	maxlist.append(no_of_letters)
   	#maxlist=maxlist+1;
   if(minlist[2]>no_of_letters):
   	minlist[2]=no_of_letters
   if(maxlist[2]<no_of_letters):
   	maxlist[2]=no_of_letters

   list.append(no_of_white_spaces)
   if(flag==False):	
   	minlist.append(no_of_white_spaces)
   	#minindex=minindex+1;
   	maxlist.append(no_of_white_spaces)
   	#maxlist=maxlist+1;
   if(minlist[3]>no_of_white_spaces):
   	minlist[3]=no_of_white_spaces
   if(maxlist[3]<no_of_white_spaces):
   	maxlist[3]=no_of_white_spaces

   list.append(no_of_upper_characters)
   if(flag==False):
   	minlist.append(no_of_upper_characters)
   	#minindex=minindex+1;
   	maxlist.append(no_of_upper_characters)
   	#maxlist=maxlist+1;
   if(minlist[4]>no_of_upper_characters):
   	minlist[4]=no_of_upper_characters
   if(maxlist[4]<no_of_upper_characters):
   	maxlist[4]=no_of_upper_characters

   list.append(no_of_tabspace_characters)
   if(flag==False):
   	minlist.append(no_of_tabspace_characters)
   	#minindex=minindex+1;
   	maxlist.append(no_of_tabspace_characters)
   	#maxlist=maxlist+1;
   if(minlist[5]>no_of_tabspace_characters):
   	minlist[5]=no_of_tabspace_characters
   if(maxlist[5]<no_of_tabspace_characters):
   	maxlist[5]=no_of_tabspace_characters

   list.append(no_of_special_characters)
   if(flag==False):
   	minlist.append(no_of_special_characters)
   	#minindex=minindex+1;
   	maxlist.append(no_of_special_characters)
   	#maxlist=maxlist+1;
   if(minlist[6]>no_of_special_characters):
   	minlist[6]=no_of_special_characters
   if(maxlist[6]<no_of_special_characters):
   	maxlist[6]=no_of_special_characters

   
   list.append(no_of_words)
   if(flag==False):
   	minlist.append(no_of_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_words)
   	#maxlist=maxlist+1;
   if(minlist[7]>no_of_words):
   	minlist[7]=no_of_words
   if(maxlist[7]<no_of_words):
   	maxlist[7]=no_of_words

   list.append(avg_length_word)
   if(flag==False):
   	minlist.append(avg_length_word)
   	#minindex=minindex+1;
   	maxlist.append(avg_length_word)
   	#maxlist=maxlist+1;
   if(minlist[8]>avg_length_word):
   	minlist[8]=avg_length_word
   if(maxlist[8]<avg_length_word):
   	maxlist[8]=avg_length_word

   list.append(no_of_unique_words)
   if(flag==False):
   	minlist.append(no_of_unique_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_unique_words)
   	#maxlist=maxlist+1;
   if(minlist[9]>no_of_unique_words):
   	minlist[9]=no_of_unique_words
   if(maxlist[9]<no_of_unique_words):
   	maxlist[9]=no_of_unique_words

   	
   list.append(no_of_words_length_morethan_six)
   if(flag==False):
   	minlist.append(no_of_words_length_morethan_six)
   	#minindex=minindex+1;
   	maxlist.append(no_of_words_length_morethan_six)
   	#maxlist=maxlist+1;
   if(minlist[10]>no_of_words_length_morethan_six):
   	minlist[10]=no_of_words_length_morethan_six
   if(maxlist[10]<no_of_words_length_morethan_six):
   	maxlist[10]=no_of_words_length_morethan_six
   	
   list.append(no_of_words_length_lessthan_three)
   if(flag==False):
   	minlist.append(no_of_words_length_lessthan_three)
   	#minindex=minindex+1;
   	maxlist.append(no_of_words_length_lessthan_three)
   	#maxlist=maxlist+1;
   if(minlist[11]>no_of_words_length_lessthan_three):
   	minlist[11]=no_of_words_length_lessthan_three
   if(maxlist[11]<no_of_words_length_lessthan_three):
   	maxlist[11]=no_of_words_length_lessthan_three
   
   list.append(no_of_single_quotes);
   if(flag==False):
   	minlist.append(no_of_single_quotes)
   	#minindex=minindex+1;
   	maxlist.append(no_of_single_quotes)
   	#maxlist=maxlist+1;
   if(minlist[12]>no_of_single_quotes):
   	minlist[12]=no_of_single_quotes
   if(maxlist[12]<no_of_single_quotes):
   	maxlist[12]=no_of_single_quotes

   list.append(no_of_commas)
   if(flag==False):
   	minlist.append(no_of_commas)
   	#minindex=minindex+1;
   	maxlist.append(no_of_commas)
   	#maxlist=maxlist+1;
   if(minlist[13]>no_of_commas):
   	minlist[13]=no_of_commas
   if(maxlist[13]<no_of_commas):
   	maxlist[13]=no_of_commas

   list.append(no_of_periods)
   if(flag==False):
   	minlist.append(no_of_periods)
   	#minindex=minindex+1;
   	maxlist.append(no_of_periods)
   	#maxlist=maxlist+1;
   if(minlist[14]>no_of_periods):
   	minlist[14]=no_of_periods
   if(maxlist[14]<no_of_periods):
   	maxlist[14]=no_of_periods

   list.append(no_of_colons)
   if(flag==False):
   	minlist.append(no_of_colons)
   	#minindex=minindex+1;
   	maxlist.append(no_of_colons)
   	#maxlist=maxlist+1;
   if(minlist[15]>no_of_colons):
   	minlist[15]=no_of_colons
   if(maxlist[15]<no_of_colons):
   	maxlist[15]=no_of_colons

   list.append(no_of_semi_colons)
   if(flag==False):
   	minlist.append(no_of_semi_colons)
   	#minindex=minindex+1;
   	maxlist.append(no_of_semi_colons)
   	#maxlist=maxlist+1;
   if(minlist[16]>no_of_semi_colons):
   	minlist[16]=no_of_semi_colons
   if(maxlist[16]<no_of_semi_colons):
   	maxlist[16]=no_of_semi_colons

   list.append(no_of_question_marks)
   if(flag==False):
   	minlist.append(no_of_question_marks)
   	#minindex=minindex+1;
   	maxlist.append(no_of_question_marks)
   	#maxlist=maxlist+1;
   if(minlist[17]>no_of_question_marks):
   	minlist[17]=no_of_question_marks
   if(maxlist[17]<no_of_question_marks):
   	maxlist[17]=no_of_question_marks

   list.append(no_of_multiple_questions_marks)
   if(flag==False):
   	minlist.append(no_of_multiple_questions_marks)
   	#minindex=minindex+1;
   	maxlist.append(no_of_multiple_questions_marks)
   	#maxlist=maxlist+1;
   if(minlist[18]>no_of_multiple_questions_marks):
   	minlist[18]=no_of_multiple_questions_marks
   if(maxlist[18]<no_of_multiple_questions_marks):
   	maxlist[18]=no_of_multiple_questions_marks

   list.append(no_of_exclamation_marks)
   if(flag==False):
   	minlist.append(no_of_exclamation_marks)
   	#minindex=minindex+1;
   	maxlist.append(no_of_exclamation_marks)
   	#maxlist=maxlist+1;
   if(minlist[19]>no_of_exclamation_marks):
   	minlist[19]=no_of_exclamation_marks
   if(maxlist[19]<no_of_exclamation_marks):
   	maxlist[19]=no_of_exclamation_marks

   list.append(no_of_multiple_exclamation)
   if(flag==False):
   	minlist.append(no_of_multiple_exclamation)
   	#minindex=minindex+1;
   	maxlist.append(no_of_multiple_exclamation)
   	#maxlist=maxlist+1;
   if(minlist[20]>no_of_multiple_exclamation):
   	minlist[20]=no_of_multiple_exclamation
   if(maxlist[20]<no_of_multiple_exclamation):
   	maxlist[20]=no_of_multiple_exclamation

   list.append(no_of_ellipsis)
   if(flag==False):
   	minlist.append(no_of_ellipsis)
   	#minindex=minindex+1;
   	maxlist.append(no_of_ellipsis)
   	#maxlist=maxlist+1;
   if(minlist[21]>no_of_ellipsis):
   	minlist[21]=no_of_ellipsis
   if(maxlist[21]<no_of_ellipsis):
   	maxlist[21]=no_of_ellipsis

   list.append(no_of_article_words)
   if(flag==False):
   	minlist.append(no_of_article_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_article_words)
   	#maxlist=maxlist+1;
   if(minlist[22]>no_of_article_words):
   	minlist[22]=no_of_article_words
   if(maxlist[22]<no_of_article_words):
   	maxlist[22]=no_of_article_words

   list.append(no_of_Adposition_words)
   if(flag==False):
   	minlist.append(no_of_Adposition_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_Adposition_words)
   	#maxlist=maxlist+1;
   if(minlist[23]>no_of_Adposition_words):
   	minlist[23]=no_of_Adposition_words
   if(maxlist[23]<no_of_Adposition_words):
   	maxlist[23]=no_of_Adposition_words

   list.append(no_of_conjunction_words)
   if(flag==False):
   	minlist.append(no_of_conjunction_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_conjunction_words)
   	#maxlist=maxlist+1;
   if(minlist[24]>no_of_conjunction_words):
   	minlist[24]=no_of_conjunction_words
   if(maxlist[24]<no_of_conjunction_words):
   	maxlist[24]=no_of_conjunction_words

   list.append(no_of_proposition_words)
   if(flag==False):
   	minlist.append(no_of_proposition_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_proposition_words)
   	#maxlist=maxlist+1;
   if(minlist[25]>no_of_proposition_words):
   	minlist[25]=no_of_proposition_words
   if(maxlist[25]<no_of_proposition_words):
   	maxlist[25]=no_of_proposition_words

   list.append(no_of_prosentence_words)
   if(flag==False):
   	minlist.append(no_of_prosentence_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_prosentence_words)
   	#maxlist=maxlist+1;
   if(minlist[26]>no_of_prosentence_words):
   	minlist[26]=no_of_prosentence_words
   if(maxlist[26]<no_of_prosentence_words):
   	maxlist[26]=no_of_prosentence_words

   list.append(no_of_auxilary_verbs)
   if(flag==False):
   	minlist.append(no_of_auxilary_verbs)
   	#minindex=minindex+1;
   	maxlist.append(no_of_auxilary_verbs)
   	#maxlist=maxlist+1;
   if(minlist[27]>no_of_auxilary_verbs):
   	minlist[27]=no_of_auxilary_verbs
   if(maxlist[27]<no_of_auxilary_verbs):
   	maxlist[27]=no_of_auxilary_verbs

   list.append(no_of_interjection_words)
   if(flag==False):
   	minlist.append(no_of_interjection_words)
   	#minindex=minindex+1;
   	maxlist.append(no_of_interjection_words)
   	#maxlist=maxlist+1;
   if(minlist[28]>no_of_interjection_words):
   	minlist[28]=no_of_interjection_words
   if(maxlist[28]<no_of_interjection_words):
   	maxlist[28]=no_of_interjection_words


def wordstem(entry):
   return filter(lambda w: len(w) > 0,[w.strip("0123456789!:,.?(){}[]") for w in entry.split()])

def wordparser(fileName):
   global flag
   f = open(fileName)
   xml = f.read()
   xml = xml.replace('\n', '').replace('\r', '')
   post_list = xml.split('<date>')
   blog_list = []
   for post in post_list:
     try:
       date = post.split('</date>')[0]
       post = post.split('<post>')[1].split('</post>')[0].strip()
       blog_list.append({'post': post})
     except:
       pass
   #word based features
   Hapax_legomena_ratio=0       # no of words occur exactly once / total no of words
   Hapax_dislegomena_ratio=0    # no of words occur exactly twice / total no of words
   Yulesk_measure=0
   Sichel_Smeasure=0
   Honores_Rmeasure=0
   Simpson_Dmeasure=0
   Entropy_measure=0
   
   #Structural Features
   no_of_paragraphs=len(blog_list)
   no_of_lines=0
   no_of_sentences=0
   no_of_sentences_upper=0
   no_of_sentences_lower=0
   avg_no_of_words_par=0
   avg_no_of_characters_par=0
   avg_no_of_sentences_par=0
   avg_no_of_words_sen=0

   d = {}
   stemmer = PorterStemmer()
   
   for i in range(len(blog_list)):
      blog_list[i]['post']=unicode(repr(blog_list[i]['post']))
      #no_of_lines+=blog_list[i]['post'].count('\n')
      sents=nltk.sent_tokenize(blog_list[i]['post'])
      no_of_sentences+=len(sents)
      text=nltk.word_tokenize(blog_list[i]['post'])
      avg_no_of_words_par+=len(text)
      avg_no_of_characters_par+=blog_list[i]['post'].count('')
      
      #count no of sentences begin with uppercase and also begin with lower case
      for i in sents:
         if i[0].isupper():
            no_of_sentences_upper+=1
         elif i[0].islower():
            no_of_sentences_lower+=1
      
      for word in text:
         word=stemmer.stem(word).lower()
         if word in d:  
                d[word] += 1  
         else:  
                d[word] = 1

      #find the frequnecy of each word
      for word in text:
         if word in uniquewords:  
                uniquewords[word] += 1  
         else:  
                uniquewords[word] = 1
   
   unique_count = 0
   double_count=0
   for each in uniquewords:  
      if uniquewords[each] == 1:  
         unique_count += 1
      if uniquewords[each]==2:
         double_count+=1
   
   freq={}
   M2=0
   no_of_words=float(avg_no_of_words_par)
   for key,value in groupby(sorted(d.values())):
      for v in value:
        if v in freq:
           freq[key]+=1
        else:
           freq[key]=1
      M2+=freq[key]*(key/no_of_words)**2
      Simpson_Dmeasure+=(freq[key]*(key/no_of_words)*(key-1)/(no_of_words-1))
      Entropy_measure+=(freq[key]*(-math.log10(key/no_of_words))*(key/no_of_words))
   
   avg_no_of_words_par=round(avg_no_of_words_par/float(no_of_paragraphs),6)
   avg_no_of_words_sen=round(avg_no_of_words_par/float(no_of_sentences),6)
   avg_no_of_characters_par=round(avg_no_of_characters_par/float(no_of_paragraphs),6)
   avg_no_of_sentences_par=round(no_of_sentences/float(no_of_paragraphs),6)
   no_of_sentences_upper=round(no_of_sentences_upper/float(no_of_sentences),6)
   no_of_sentences_lower=round(no_of_sentences_lower/float(no_of_sentences),6)

   Hapax_legomena_ratio = round(unique_count/no_of_words,6)
   Hapax_dislegomena_ratio=round(double_count/no_of_words,6)
   Yulesk_measure=round((10**4)*((-1/no_of_words)+M2),6)
   Sichel_Smeasure=round(double_count/float(len(words)),6)
   Honores_Rmeasure=round((100*math.log10(no_of_words))/(1-(unique_count/len(words))),6)
   
   list.append(avg_no_of_words_par)
   if(flag==False):
      minlist.append(avg_no_of_words_par)
      #minindex=minindex+1;
      maxlist.append(avg_no_of_words_par)
      #maxlist=maxlist+1;
   if(minlist[29]>avg_no_of_words_par):
   	minlist[29]=avg_no_of_words_par
   if(maxlist[29]<avg_no_of_words_par):
   	maxlist[29]=avg_no_of_words_par

   list.append(avg_no_of_words_sen)
   if(flag==False):
   	minlist.append(avg_no_of_words_sen)
   	#minindex=minindex+1;
   	maxlist.append(avg_no_of_words_sen)
   	#maxlist=maxlist+1;
   if(minlist[30]>avg_no_of_words_sen):
   	minlist[30]=avg_no_of_words_sen
   if(maxlist[30]<avg_no_of_words_sen):
   	maxlist[30]=avg_no_of_words_sen

   list.append(avg_no_of_characters_par)
   if(flag==False):
   	minlist.append(avg_no_of_characters_par)
   	#minindex=minindex+1;
   	maxlist.append(avg_no_of_characters_par)
   	#maxlist=maxlist+1;
   if(minlist[31]>avg_no_of_characters_par):
   	minlist[31]=avg_no_of_characters_par
   if(maxlist[31]<avg_no_of_characters_par):
   	maxlist[31]=avg_no_of_characters_par

   list.append(avg_no_of_sentences_par)
   if(flag==False):
   	minlist.append(avg_no_of_sentences_par)
   	#minindex=minindex+1;
   	maxlist.append(avg_no_of_sentences_par)
   	#maxlist=maxlist+1;
   if(minlist[32]>avg_no_of_sentences_par):
   	minlist[32]=avg_no_of_sentences_par
   if(maxlist[32]<avg_no_of_sentences_par):
   	maxlist[32]=avg_no_of_sentences_par

   list.append(no_of_sentences_upper)
   if(flag==False):
   	minlist.append(no_of_sentences_upper)
   	#minindex=minindex+1;
   	maxlist.append(no_of_sentences_upper)
   	#maxlist=maxlist+1;
   if(minlist[33]>no_of_sentences_upper):
   	minlist[33]=no_of_sentences_upper
   if(maxlist[33]<no_of_sentences_upper):
   	maxlist[33]=no_of_sentences_upper

   list.append(no_of_sentences_lower)
   if(flag==False):
   	minlist.append(no_of_sentences_lower)
   	#minindex=minindex+1;
   	maxlist.append(no_of_sentences_lower)
   	#maxlist=maxlist+1;
   if(minlist[34]>no_of_sentences_lower):
   	minlist[34]=no_of_sentences_lower
   if(maxlist[34]<no_of_sentences_lower):
   	maxlist[34]=no_of_sentences_lower

   list.append(Hapax_legomena_ratio)
   if(flag==False):
   	minlist.append(Hapax_legomena_ratio)
   	#minindex=minindex+1;
   	maxlist.append(Hapax_legomena_ratio)
   	#maxlist=maxlist+1;
   if(minlist[35]>Hapax_legomena_ratio):
   	minlist[35]=Hapax_legomena_ratio
   if(maxlist[35]<Hapax_legomena_ratio):
   	maxlist[35]=Hapax_legomena_ratio

   list.append(Hapax_dislegomena_ratio)
   if(flag==False):
   	minlist.append(Hapax_dislegomena_ratio)
   	#minindex=minindex+1;
   	maxlist.append(Hapax_dislegomena_ratio)
   	#maxlist=maxlist+1;
   if(minlist[36]>Hapax_dislegomena_ratio):
   	minlist[36]=Hapax_dislegomena_ratio
   if(maxlist[36]<Hapax_dislegomena_ratio):
   	maxlist[36]=Hapax_dislegomena_ratio

   list.append(Yulesk_measure)
   if(flag==False):
   	minlist.append(Yulesk_measure)
   	#minindex=minindex+1;
   	maxlist.append(Yulesk_measure)
   	#maxlist=maxlist+1;
   if(minlist[37]>Yulesk_measure):
   	minlist[37]=Yulesk_measure
   if(maxlist[37]<Yulesk_measure):
   	maxlist[37]=Yulesk_measure

   list.append(Sichel_Smeasure)
   if(flag==False):
   	minlist.append(Sichel_Smeasure)
   	#minindex=minindex+1;
   	maxlist.append(Sichel_Smeasure)
   	#maxlist=maxlist+1;
   if(minlist[38]>Sichel_Smeasure):
   	minlist[38]=Sichel_Smeasure
   if(maxlist[38]<Sichel_Smeasure):
   	maxlist[38]=Sichel_Smeasure

   list.append(Honores_Rmeasure)
   if(flag==False):
   	minlist.append(Honores_Rmeasure)
   	#minindex=minindex+1;
   	maxlist.append(Honores_Rmeasure)
   	#maxlist=maxlist+1;
   if(minlist[39]>Honores_Rmeasure):
   	minlist[39]=Honores_Rmeasure
   if(maxlist[39]<Honores_Rmeasure):
   	maxlist[39]=Honores_Rmeasure

   flag=True
   #print Simpson_Dmeasure,Entropy_measure,Hapax_legomena_ratio,Hapax_dislegomena_ratio,Sichel_Smeasure,Honores_Rmeasure,Yulesk_measure
   #print no_of_lines
   #print no_of_sentences_lower
   #print no_of_paragraphs,no_of_sentences,avg_no_of_words_par,avg_no_of_characters_par,avg_no_of_sentences_par,avg_no_of_words_sen
   #print no_of_sentences
   

'''print no_of_words print no_of_article_words print no_of_Adposition_words
print no_of_conjunction_words
print no_of_proposition_words
print no_of_single_quotes,no_of_commas,no_of_periods,no_of_colons,no_of_semi_colons,no_of_question_marks,no_of_multiple_questions_marks,no_of_exclamation_marks,no_of_multiple_exclamation,no_of_ellipsis
'''
import os
import re

for file in os.listdir("trainblogs"):
  if file.endswith(".xml"):
     list=[]
     words=set()
     uniquewords = dict()
     if re.search('\.male',file):
        classes.append(0)
     else:
        classes.append(1)
     parse("trainblogs/"+file)
     wordparser("trainblogs/"+file)
     samplelist.append(list)
     #print samplelist
     #print classes
'''samplelist=[[4891, 0.002045, 0.745451, 0.199141, 0.034553, 0.0, 1094.0, 0.108775, 0.563071, 0.004703, 0.007156, 0.011041, 0.000613, 0.0, 0.0, 0.0, 0.00368, 0.0, 0.000613, 0.077697, 0.090494, 0.034735, 0.109689], [23660, 0.003212, 0.783897, 0.17836, 0.026331, 0.0, 4647.0, 0.200129, 0.490424, 0.004438, 0.008199, 0.008199, 0.00038, 0.001817, 0.000761, 0.0, 0.000254, 0.0, 4.2e-05, 0.082849, 0.092102, 0.039595, 0.067355], [28867, 0.006028, 0.76319, 0.181245, 0.042471, 0.0, 6165.0, 0.151663, 0.480616, 0.003256, 0.008522, 0.01881, 0.00052, 0.00097, 0.000797, 0.0, 0.001282, 0.0, 0.000173, 0.053528, 0.063909, 0.027575, 0.033252], [74934, 0.002576, 0.7653, 0.18934, 0.038621, 0.0, 16796.0, 0.124315, 0.547095, 0.00762, 0.008901, 0.016401, 0.00056, 0.000107, 0.000694, 0.0, 0.001975, 0.0, 0.001655, 0.076506, 0.085497, 0.027983, 0.089783], [37704, 0.019308, 0.739789, 0.197592, 0.010185, 0.0, 8600.0, 0.098488, 0.535814, 0.001936, 0.011776, 0.016099, 0.000902, 2.7e-05, 0.001087, 5.3e-05, 0.001804, 0.000159, 0.000504, 0.058953, 0.078256, 0.040698, 0.075581], [5155, 0.001552, 0.757323, 0.19709, 0.037633, 0.0, 1152.0, 0.128472, 0.559028, 0.005432, 0.010475, 0.014549, 0.000776, 0.0, 0.000194, 0.0, 0.005626, 0.0, 0.000194, 0.088542, 0.075521, 0.020833, 0.06684], [14619, 0.001573, 0.767016, 0.189001, 0.031534, 0.0, 3109.0, 0.145706, 0.522676, 0.00684, 0.010397, 0.015323, 0.000889, 0.0, 0.001163, 0.0, 0.000821, 0.0, 0.001368, 0.07816, 0.095529, 0.023159, 0.078803], [40850, 0.00022, 0.782938, 0.185581, 0.020979, 0.0, 8718.0, 0.146823, 0.518009, 0.00049, 0.012729, 0.009963, 0.0, 0.002424, 0.000392, 0.0, 0.0, 0.0, 0.0, 0.09532, 0.110117, 0.039803, 0.100252], [3930, 0.002036, 0.762595, 0.201018, 0.030789, 0.0, 868.0, 0.110599, 0.526498, 0.005598, 0.008906, 0.011705, 0.000254, 0.0, 0.001272, 0.0, 0.00229, 0.0, 0.0, 0.09447, 0.095622, 0.016129, 0.084101], [12585, 0.004132, 0.767263, 0.185459, 0.029718, 0.0, 2761.0, 0.125317, 0.532054, 0.009217, 0.005721, 0.012157, 0.000397, 0.0, 0.00143, 0.0, 0.000477, 7.9e-05, 0.001351, 0.083303, 0.086563, 0.03477, 0.069178], [19623, 0.002956, 0.7772, 0.181573, 0.032972, 0.0, 3900.0, 0.201795, 0.496667, 0.003261, 0.006064, 0.013148, 0.000306, 0.000917, 0.000357, 5.1e-05, 0.000408, 0.0, 0.00107, 0.093846, 0.098205, 0.035385, 0.046667], [45138, 0.005162, 0.775909, 0.180956, 0.053547, 0.000576, 9057.0, 0.205918, 0.49067, 0.005782, 0.007222, 0.012008, 0.001351, 6.6e-05, 0.000377, 0.0, 0.000665, 0.0, 2.2e-05, 0.081153, 0.099039, 0.022855, 0.050569]]
#print samplelist
classes=[1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0]'''

# minlist
#print "\n"
#print maxlist
#print "\n"
#print samplelist

#writing features to text file
fp=open('training1.txt','w')

for i in range(len(samplelist)):
   for j in range(len(samplelist[i])):
      if maxlist[j]!=minlist[j]:
         samplelist[i][j]=round((samplelist[i][j]-minlist[j])/(maxlist[j]-minlist[j]),6)

for i in range(len(samplelist)):
   st=str(samplelist[i]).replace("[",'').replace("]",'')
   fp.write("".join(st+","+str(classes[i])+"\n"))
fp.close()
'''

#reading from training data
samplelist=[]
classes=[]
fp=open('training.txt','r')
lists=fp.readlines()
for line in lists:
   list=[]
   line=line.strip()
   features=line.split(",")
   classes.append(int(features[len(features)-1]))
   for i in range(len(features)-1):
      list.append(float(features[i]))
   samplelist.append(list)


#apply SVM to fit the features with class lables
clf=svm.SVC()
clf.fit(samplelist,classes)

#print classes
testlist=[]
y_test=[]


#loading for test data
for file in os.listdir("testblog"):
   
   if file.endswith(".xml"):
      list=[]
      if re.search('\.male',file):
        y_test.append(0)
      else:
        y_test.append(1)
      parse("testblog/"+file)
      wordparser("testblog/"+file)
      testlist.append(list)

for i in range(len(testlist)):
   for j in range(len(testlist[i])):
      if maxlist[j]!=minlist[j]:
         testlist[i][j]=round((testlist[i][j]-minlist[j])/(maxlist[j]-minlist[j]),6)

fp1=open('testing.txt','r')
lists=fp.readlines()
for line in lists:
   list=[]
   line=line.strip()
   features=line.split(",")
   y_test.append(int(features[len(features)-1]))
   for i in range(len(features)-1):
      list.append(float(features[i]))
   testlist.append(list)

#print testlist
#print clf.predict(trainlist)
y_true, y_pred = y_test, clf.predict(testlist)
acc = len(numpy.where(y_true==y_pred)[0])/float(len(y_true))*100
print "average accuracy",acc
print(classification_report(y_true, y_pred))
print y_true
print y_pred
