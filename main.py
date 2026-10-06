import numpy as np
import weakref
import contextlib

class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.name = name # ex) x = Variable(np.array(1.0), 'input_x')
        self.grad = None # Variable 본인이 dx일 때의 grad 값을 기록하는 변수 
        self.creator = None # 부모 함수 기록하는 변수
        self.generation = 0 # 역전파시 순서 꼬임을 방지하기 위해서 순전파 진행시에 세대 수를 기록하는 변수

    def set_creator(self, func):
        self.creator = func 
        self.generation = func.generation + 1 # input (a 세대) -> func (a세대) -> variable(a+1세대)

    def backward(self, retain_grad=False): # 중간 변수에 대해서는 미분값을 제거할 것 -> FLAG
        if self.grad is None:
            self.grad = np.ones_like(self.data) # backward method 간소화 

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set: # seen_set ? 
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # x, y = f.input, f.output
            # x.grad = f.backward(y.grad)

            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad: # False면 미분값 유지 X <- 메모리 효율성 증대
                for y in f.outputs:
                    y().grad = None # y는 약한 참조 (weakref), 구하고자 하는 값, dx(말단값)이 아니면 weakref을 통해서 없애버림

    def cleargrad(self):
        self.grad = None

    @property # shape라는 메소드를 인스턴스 변수처럼 활용 가능 // ex) print(Variable.shape)
    def shape(self):
        return self.data.shape

    @property
    def ndim(self): # n of dim
        return self.data.ndim

    @property
    def size(self): # n of elements
        return self.data.size

    @property
    def dtype(self): # type of data
        return self.data.dtype

    def __len__(self): # len은 파이썬의 내장 메서드, __을 붙여 특수 메서드로 바꾼 후 Variable 인스턴스에 대해서도 len 함수 사용 가능
        return len(self.data) # print(len(x))

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'Variable(' + p + ')'

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
        
    
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) # x0(3세대), x1(4세대) -> f(4세대) <<세대 설정>>

            for output in outputs:
                output.set_creator(self) # 연결 설정
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Square(Function): # y = x^2
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * 2 * x
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.exp(x)
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1 
        return y

    def backward(self, gy):
        return gy, gy

class Product(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

class Divide(Function):
    def forward(self, x0, x1):
        if(x1 == 0):
            raise ZeroDivisionError()
        y = x0 / x1
        return y

class Config:
    enable_backprop = True # True -> 역전파 활성 모드
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def product(x0, x1):
    return Product()(x0, x1)

def divide(x0, x1):
    return Divide()(x0, x1)


def num_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

@contextlib.contextmanager # try 전 : 전처리, try 후 : 후처리, 실행중 <-- with 안의 로직 실행
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

# 역전파가 필요 없는 경우에는 no_grad 호출 시 함수를 적으면 됨

# ---------- example ----------

# with no_grad():
#   x = Variable(np.array(2.0))
#   y = square(x)


# -------------------------------------------------------------------------
