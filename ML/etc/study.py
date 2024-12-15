
# 값 교환(tuple)
a, b = b, a

# 드모르간의 법칙
# x and y = not(not x or not y)
# x or y = not(not x and not y)

# list to tuple
x = [1, 2, 3]
a, b, c = x
a, b, c
# (1, 2, 3)

# a = b = 1(X) ; '='는 연산자가 아니기 때문

# True
[1,2,3] < [1,2,4]
[1,2,3] < [1,2,3,4]

# 등가성(equality), 동일성(identity)
# 등가성(값이 같은가? '=='), 동일성 (동일한 객체인가? 'is')

# list 생성
New = [None] * 10

# 리스트 참조
'''
lst1 = [1,2,3,4,5]
lst2 = lst1 #list참조
lst1 is lst2 -> true
lst1[2] = 9
lst1
[1,2,9,4,5]
lst2
[1,2,9,4,5]
'''

# x = ['John', 'George']
for i, name in enumerate(x, 1): #1번부터 출력
    print(f'{i}번째 = {name}')

# list 역순 정렬 [].reverse()

# 진수변환 : 리스트 문자열 -> 해당문자를 꺼내 결합

# import copy
# x.copy(얕은 복사), x.deepcopy(깊은복사)

# 검색
# 배열 검색, 연결 리스트 검색, 이진 검색 트리 검색

# 선형 검색
from typing import Any, Sequence
def seq_search(a: Sequence, key : Any) -> int:
    # Sequence a에서 key와 값이 같은 원소를 선형 검색(for문)
    i = 0
    while True:
        if i == len(a):
            return -1
        if a[i] == key:
            return i
        i+=1


# 보초법(sentinel method)
'검색하고자 하는 key값을 배열 끝에 저장하여 종료조건을 동일하게 하여 코드를 간단히함'

# 이진 검색



'즐겁게 일할수 있는 사람과 일하라'
'존경하지 않는 사람을 위해 일하지말라 - 능력, 인품'
'당신 스스로도 사지않을걸 팔지마라 - 다른 사람의 시간을 낭비시키지마라'