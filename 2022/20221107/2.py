# 큰수의 법칙
# 첫째줄에 n,m,k가 주어짐
# 둘째줄에 n개의 자연수가 주어짐 (자연수는 공백으로 주어짐)
# 입력으로 주어지는 k는 항상 m 보다 작거나 같다
# 주어진 수들을 m번 더하여 가장 큰 수를 만들기 (단, 배열의 특정 숫자가 k번을 초과하여 더해질 수 없음)
# 아이디어 : 가장 큰수를 k 번 더하고 사이에 그다음으로 작은 수를 넣자
import sys

def solution(inputs):
    n,m,k = map(int, inputs[0].split())
    numbers = list(inputs[1].split())
    numbers = [int(i) for i in numbers]
    # numbers = [for list in lists]
    print(numbers)
    print(n,m,k)
    sorted_numbers = sorted(numbers, reverse = True)
    f = sorted_numbers[0]
    s = sorted_numbers[1]
    result = f*k + s
    answer = result * (m//(k+1)) + f * m%(k+1)
    return answer