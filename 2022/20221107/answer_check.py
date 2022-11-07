# 파일 실행시 첫번째 인자로 파일명 입력해야함 (확장자 제외하고 파일명만 입력)
# 같은 폴더 내에 답안(py), 문제+정답 (txt) 파일 2개가 모두 존재해야 함.
# 문제+정답 파일의 가장 마지막 줄에 답을 입력해야함. 
# 답안은 solution 함수의 반환값으로 제출해야함.

import sys
import importlib

assert len(sys.argv) >= 2, '첫번째 인자로 파일명을 입력하세요.'
file = sys.argv[1]
module = importlib.import_module(file)

with open(file+'.txt') as file_data:
    data = [item.strip() for item in list(file_data)]

inputs = data[:-1]
answer = data[-1]
your_answer = str(module.solution(inputs))
assert answer == your_answer, f'정답이 틀렸습니다. answer = {answer} your answer = {your_answer}'
print(f'정답이 일치합니다. answer = {answer} your answer = {your_answer}')


