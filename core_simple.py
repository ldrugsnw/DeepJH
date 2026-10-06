import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True  # True -> 역전파 활성 모드


@contextlib.contextmanager
# try 전: 설정 변경, try 후: 원래 설정 복구
# with 블록 안의 로직이 실행되는 동안 설정이 유지됨
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


# 역전파가 필요 없는 경우 no_grad() 사용
# with no_grad():
#     x = Variable(np.array(2.0))
#     y = square(x)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name  # ex) x = Variable(np.array(1.0), 'input_x')
        self.grad = None  # Variable의 기울기를 기록하는 변수
        self.creator = None  # 부모 함수 기록
        self.generation = 0  # 역전파 순서가 꼬이지 않도록 세대 기록

    @property
    # shape 메서드를 인스턴스 속성처럼 활용 가능
    # ex) print(x.shape)
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):  # number of dimensions
        return self.data.ndim

    @property
    def size(self):  # number of elements
        return self.data.size

    @property
    def dtype(self):  # type of data
        return self.data.dtype

    def __len__(self):
        # __len__ 특수 메서드를 통해 Variable에도 len() 사용 가능
        # ex) print(len(x))
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        # input(a세대) -> func(a세대) -> variable(a+1세대)
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        # retain_grad=False이면 중간 변수의 미분값을 제거
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # backward 메서드 간소화

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                # 이미 추가한 함수를 중복 등록하지 않음
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            # 각 출력 변수의 기울기를 가져옴
            # output은 weakref이므로 output()으로 실제 객체에 접근
            gys = [output().grad for output in f.outputs]

            gxs = f.backward(*gys)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                # False이면 중간 변수의 기울기를 제거하여 메모리 절약
                # weakref 자체가 삭제하는 것은 아님
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            # x0(3세대), x1(4세대) -> f(4세대)
            # 입력 중 가장 높은 세대를 함수의 세대로 설정
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)  # 연결 설정

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


# Operator Overloading
# 파이썬 연산자를 Variable에서도 사용할 수 있도록 설정
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add

    Variable.__mul__ = mul
    Variable.__rmul__ = mul

    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub

    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
