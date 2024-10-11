import pathlib
import copy
def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A=[]
    B=[]
    file= open(path, 'r')
    for line in file:
        ROW=[]
        print(line)
        terms=line.split(' ')
        last_sign = 1

        for term in terms:
            if term == '+':
                last_sign = 1
            elif term == '-':
                last_sign = -1
            elif 'x' in term or 'y' in term or 'z' in term:
                if 'x' in term:
                    coeff = term.replace('x', '')
                    coeff = coeff.strip()
                    if coeff == '':
                        coeff = '1'
                    coeff = float(coeff)
                    ROW.append(last_sign * float(coeff))
                elif 'y' in term:
                    coeff = term.replace('y', '')
                    coeff = coeff.strip()
                    if coeff == '':
                        coeff = '1'
                    coeff = float(coeff)
                    ROW.append(last_sign * float(coeff))
                elif 'z' in term:
                    coeff = term.replace('z', '')
                    coeff = coeff.strip()
                    if coeff == '':
                        coeff = '1'
                    coeff = float(coeff)
                    ROW.append(last_sign * float(coeff))
                last_sign = 1
        constant = float(terms[-1].strip())
        B.append(constant)
        A.append(ROW)
    return A, B

A, B = load_system(pathlib.Path("D:\\facultate\\Anul_3\\semestrul 1\\retele_neuronale\\Neural-Networks-Template-2024\\assiignment_1\\input.txt"))
print(f"{A=} {B=}")


def determinant(matrix: list[list[float]]) -> float:
    determinant=0
    first_term=matrix[0][0]*(matrix[1][1]*matrix[2][2] - matrix[1][2]*matrix[2][1])
    second_term=matrix[0][1]*(matrix[1][0]*matrix[2][2] - matrix[1][2]*matrix[2][0])
    third_term=matrix[0][2]*(matrix[1][0]*matrix[2][1] - matrix[1][1]*matrix[2][0])

    determinant=first_term-second_term+third_term
    return determinant

def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0]+matrix[1][1]+matrix[2][2]

print(f"{determinant(A)=}")
print(f"{trace(A)=}")

def norm(vector: list[float]) -> float:
    return (vector[0]**2 + vector[1]**2 + vector[2]**2)**0.5

print(f"{norm(B)=}")

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    transpose=[[],[],[]]
    transpose[0]=[matrix[0][0],matrix[1][0],matrix[2][0]]
    transpose[1]=[matrix[0][1],matrix[1][1],matrix[2][1]]
    transpose[2]=[matrix[0][2],matrix[1][2],matrix[2][2]]
    return transpose

print(f"{transpose(A)=}")


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    result=[]
    for line in matrix:
        rez=0
        for i in range(0,3):
            rez=rez+line[i]*vector[i]
        result.append(rez)
    return result

print(f"{multiply(A, B)=}")
detA=determinant(A)
def replace_column(no, replacement: list[float], matrix:list[list[float]])->list[list[float]]:
    rMatrix = copy.deepcopy(matrix)
    for i in range (3):
        rMatrix[i][no]=replacement[i]
    return rMatrix
def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    print(matrix)
    matrixX=replace_column(0,vector,matrix)
    matrixY = replace_column(1, vector, matrix)
    matrixZ = replace_column(2, vector, matrix)
    print (detA)
    print(determinant(matrixX))
    print(matrixX)
    x=determinant(matrixX)/detA
    y=determinant(matrixY)/detA
    z=determinant(matrixZ)/detA
    return [x,y,z]

print(f"{solve_cramer(A, B)=}")

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    minor=[];
    #print(matrix)
    for i1 in range (0,3):
        if i1!=i:
         row=[]
         for j1 in range(0,3):
            if j1!=j:
                row.append(matrix[i1][j1])
         minor.append(row)
    return minor;

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
   result=[]
   for i in range(0,3):
       row=[]
       for j in range(0,3):
           minor_for_matrix=minor(matrix,i,j)
           determinant_minor=minor_for_matrix[0][0]*minor_for_matrix[1][1]- minor_for_matrix[0][1]*minor_for_matrix[1][0];
           if (i+j)%2==0:
               sign=1
           else:
               sign=-1
           row.append(sign*determinant_minor)
       result.append(row)
   return result

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    intermediate_matrix= transpose(cofactor(matrix))
    result=[[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for i in range (0,3):
        for j in range (0,3):
            result[i][j]=intermediate_matrix[i][j]*(1/determinant(matrix))
    print(result)
    return result
print(cofactor(A))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return multiply(adjoint(matrix),vector)

print(f"{solve(A, B)=}")