'''
    Config 是一个“可序列化、可打印、可延迟实例化、可控副作用”的对象工厂，
    用来把 args.json 里的“静态配置”变成“真实运行中的对象”。

'''
import os
import collections
import importlib
import pickle

def import_class(_class):
    '''
        打破静态导入的硬编码限制。在大型科研代码库中，我们可能频繁切换不同的网络结构（如 UNet, DiT），
        此函数允许通过字符串（如 'models.TemporalUnet'）动态加载类，使配置文件能以纯文本/JSON 形式保存并跨设备传输。

        它决定了后续 instance 到底是由哪段代码生成的。如果返回了错误的类，整个训练任务将完全偏离预期。

        若删除，原本灵活的 args.config 将失效，你必须在每个脚本里手动写死 import 语句，丧失了自动化扫参和多实验管理的能力。
    '''
    # 如果传进来的已经是一个类对象，直接返回
    if type(_class) is not str: return _class
    '''
        例如要导入diffuser/utils/render.py，
        那么repo_name是'diffuser'；
        module_name是'utils'；
        class_name是'render'。
    ''' 
    repo_name = __name__.split('.')[0]
    ## eg, 'utils'
    module_name = '.'.join(_class.split('.')[:-1])
    class_name = _class.split('.')[-1]
    module = importlib.import_module(f'{repo_name}.{module_name}')
    # 真正执行import
    # from repo_name.module_name import class_name
    _class = getattr(module, class_name)
    print(f'[ utils/config ] Imported {repo_name}.{module_name}:{class_name}')
    return _class

class Config(collections.Mapping):
    '''
        配置对象的容器。它不仅仅是字典，更是一个带有“记忆”和“实例化能力”的工厂蓝图。
        collections.Mapping 表现形式是一个嵌套的不可变字典，来打包所有的配置参数。
    '''

    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        '''
            捕获组件初始化的所有静态参数。

            此时传入的 kwargs 通常是 args.json 中定义的超参数。
            - 关键输出：如果指定了 savepath，会产出一个 pickle 文件。
                    这个文件是复现的核心，因为它比代码变更更稳定。
        '''
        self._class = import_class(_class) # 动态导入类,把字符串/类->确定的python类
        self._device = device 
        self._dict = {}

        # 参数被存入 _dict，作为后续实例化时的默认参数
        for key, val in kwargs.items():
            self._dict[key] = val

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if type(savepath) is tuple else savepath # 处理路径
            pickle.dump(self, open(savepath, 'wb')) # 序列化存盘，‘wb’以二进制写入，生成一个.pkl文件
            print(f'[ utils/config ] Saved config to: {savepath}\n')

    def __repr__(self):
        '''
            决定了你在 log 文件或控制台看到的参数列表。
            清晰的可视化能让你在实验运行初期的 10 秒内发现是否填错了某个关键超参数，节省大量无效算力。
        '''
        string = f'\n[utils/config ] Config: {self._class}\n'
        for key in sorted(self._dict.keys()):
            val = self._dict[key]
            string += f'    {key}: {val}\n'
        return string

    def __iter__(self):
        return iter(self._dict)

    def __getitem__(self, item):
        return self._dict[item]

    def __len__(self):
        return len(self._dict)

    def __getattr__(self, attr):
        '''
            提供顺滑的访问体验。允许以 args.learning_rate 形式访问，而不必使用 args['learning_rate']。
        '''
        if attr == '_dict' and '_dict' not in vars(self):
            self._dict = {}
            return self._dict
        try:
            return self._dict[attr]
        except KeyError:
            raise AttributeError(attr)

    def __call__(self, *args, **kwargs):
        '''
            这是该类的灵魂。当你调用 config() 时，会将训练脚本“临时传入的动态参数”和config中冻结的_dict参数合并，形成真正的构造函数
        '''
        instance = self._class(*args, **kwargs, **self._dict)
        if self._device:
            instance = instance.to(self._device)
        return instance
