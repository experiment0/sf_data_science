from IPython.display import display, Markdown

# Словарь с данными иерархии файлов
content_tree = {
    # Ключ - имя файла
    '1_main.ipynb': {
        # Заголовок файла
        'name': 'Главная',
        # Файлы с дочерними разделами
        'children': {
            '2_1_prepare_data.ipynb': {
                'name': 'Первичная обработка данных',
                'children': {
                        '2_2_prepare_data_newbies.ipynb': {
                            'name': 'Исследование данных таблицы `data_newbies` на пропуски и выбросы',
                        }, 
                        '2_2_prepare_data_participated.ipynb': { 
                            'name': 'Исследование данных таблицы `data_participated` на пропуски и выбросы',
                        },
                },
            },
            '3_2_eda_newbies.ipynb': {
                'name': 'Исследование данных таблицы `data_newbies`',
            },
            '3_2_eda_participated.ipynb': {
                'name': 'Исследование данных таблицы `data_participated`',
            },
            '4_2_feature_inginiring_newbies.ipynb': {
                'name': 'Отбор и преобразование признаков таблицы `data_newbies`',
            },
            '4_2_feature_inginiring_participated.ipynb': {
                'name': 'Отбор и преобразование признаков таблицы `data_participated`',
            },
            '5_2_classification_newbies.ipynb': {
                'name': 'Построение прогноза для данных из таблицы `data_newbies`',
            },
            '5_2_classification_participated.ipynb': {
                'name': 'Построение прогноза для данных из таблицы `data_participated`',
            },
        },
     },    
}

class Navigation:
    def __init__(
        self, 
        current_file: str, 
        content_tree: dict = content_tree
    ) -> None:
        """Выводит элементы навигации по файлам

        Args:
            current_file (str): имя текущего файла
            content_tree (dict, optional): словарь с данными о структуре файлов. По умолчанию content_tree.
        """
        # Имя текущего файла
        self.current_file = current_file
        # Словарь с данными иерархии файлов
        self.content_tree = content_tree
        # Родительский файл (для текущего)
        self.parent_file = self.get_parent_key(current_file)
        # Список с именами родительских файлов (для текущего)
        self.path_keys = self.get_path_keys(current_file)
        # Данные для хлебной крошки
        self.breadcbumbs_data = self.get_breadcbumbs_data()
        
    
    def get_parent_key(
        self,
        child_key: str, 
        data_tree: dict = None, 
        parent_key: str = None, 
    ) -> str:
        """Ищет ключ родительского раздела в дереве
        Args:
            child_key (str): ключ, для которого нужно найти родительский
            data_tree (dict, optional): словарь с деревом. По умолчанию None.
            parent_key (str, optional): переменная, в которой храним родительский ключ во время итераций. 
                                        По умолчанию None.
        Returns:
            str: ключ родительского раздела
        """
        if (data_tree is None):
            data_tree = self.content_tree
            
        for key, data in data_tree.items():        
            if (key == child_key):
                return parent_key
            if (isinstance(data, dict) and 'children' in data):
                result_key = self.get_parent_key(child_key, data['children'], key)
                if (result_key):
                    return result_key
                
    
    def get_path_keys(
        self,
        child_key: str, 
        data_tree: dict = None, 
        path_keys: list = None,
    ) -> list:
        """Возвращает список с ключами родительских разделов (до корневого)
        Args:
            child_key (str): ключ, для которого нужно найти все родительские
            data_tree (dict, optional): словарь с деревом. По умолчанию None.
            path_keys (list, optional): список с родительскими ключами, который дополняем в процессе итераций. 
                                        По умолчанию None.
        Returns:
            list: список с ключами родительских разделов
        """
        if (data_tree is None):
            data_tree = self.content_tree
            
        if (path_keys is None):
            path_keys = []
        
        parent_key = self.get_parent_key(child_key, data_tree)
        if (parent_key is None):
            return path_keys
        
        path_keys.insert(0, parent_key)
        return self.get_path_keys(parent_key, data_tree, path_keys)
    
    
    def get_breadcbumbs_data(self) -> list:
        """Возвращает список с данными для хлебной крошки
        Returns:
            list: список с данными для хлебной крошки
        """
        breadcbumbs_data = []
        root_file_name = self.path_keys[0]
        current_item = self.content_tree[root_file_name]
        
        for index, file_name in enumerate(self.path_keys):
            if (index > 0):
                current_item = current_item['children'][file_name]
                
            breadcbumb_item = {
                'link': f'./{file_name}',
                'name': current_item['name'],
            }
            breadcbumbs_data.append(breadcbumb_item)
        
        breadcbumbs_data.append({ 'name': current_item['children'][self.current_file]['name']})
        
        return breadcbumbs_data
    
    
    def display_header(self):
        """Выводит хлебную крошку и заголовок страницы
        """
        breadcbumbs_str = ''
        header = ''
        
        for item in self.breadcbumbs_data:
            if ('link' in item):
                breadcbumbs_str += f"[{item['name']}]({item['link']}) &raquo; "
            else:
                breadcbumbs_str += item['name']
                header = item['name']
                
        display(Markdown(breadcbumbs_str))
        display(Markdown(f'# {header}'))
        
    
    def display_backlink(self):
        """Выводит ссылку для возврата на родительский файл
        """
        backlink_data = self.breadcbumbs_data[-2]        
        backlink_str = f"**Вернуться к файлу &laquo;[{backlink_data['name']}]({backlink_data['link']})&raquo;**"    
        display(Markdown(backlink_str))