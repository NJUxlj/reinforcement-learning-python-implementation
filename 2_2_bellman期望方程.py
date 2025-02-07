import sympy
from sympy import symbols
from sympy.printing.str import StrPrinter


def pretty_print_solution(result):
    """将sympy解出的字典结果转换为易读的等式形式"""
    # 创建一个自定义的字符串打印器
    printer = StrPrinter()
    # 解析每个变量的结果  
    equations = []  
    for var, expr in result.items():  
        # 使用printer将表达式转换为字符串  
        var_str = printer.doprint(var)  
        expr_str = printer.doprint(expr)  
        # 构建等式字符串  
        equation = f"{var_str} = {expr_str}"  
        equations.append(equation)  

    return "\n".join(sorted(equations))  



sympy.init_printing()
v_hungry, v_full = symbols('v_hungry v_full')
q_hungry_eat, q_hungry_none, q_full_eat, q_full_none = \
        symbols('q_hungry_eat q_hungry_none q_full_eat q_full_none')
alpha, beta, x, y, gamma = symbols('alpha beta x y gamma')
system = sympy.Matrix((
        (1, 0, x-1, -x, 0, 0, 0),
        (0, 1, 0, 0, -y, y-1, 0),
        (-gamma, 0, 1, 0, 0, 0, -2),
        ((alpha-1)*gamma, -alpha*gamma, 0, 1, 0, 0, 4*alpha-3),
        (-beta*gamma, (beta-1)*gamma, 0, 0, 1, 0, -4*beta+2),
        (0, -gamma, 0, 0, 0, 1, 1) )) # 标准形式的系数矩阵
result = sympy.solve_linear_system(system,
        v_hungry, v_full,
        q_hungry_eat, q_hungry_none, q_full_eat, q_full_none) # 求解


print("线性方程组的解为：")
print("-" * 50)
print(pretty_print_solution(result))