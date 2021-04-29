# from sidecar import Sidecar
from IPython.display import display
from ipywidgets import Accordion, FileUpload, IntSlider, Tab, Text, Button, Dropdown, AppLayout, Box, HBox, VBox
from traitlets import default, List, HasTraits

class Dashboard(HasTraits):
    project_list = List
    
    def __init__(self):
        self.t_file_upload = FileUpload(description='Upload an image for Analysis')
        self.acc_open_files = Accordion(description='Yowza')
        self.header = HBox(children=[self.t_file_upload])
        self.center_panel = VBox(children=[self.acc_open_files])
        self.right_panel = VBox(children=[])
        
        self.app_layout = AppLayout(
            header=self.header, left_sidebar=None, center=self.center_panel, right_sidebar=self.right_panel, pane_widths=["3%", "72%", "25%"], pane_heights=["13%", "85%", "2%"], footer=None)
        self.app_tab = Tab(children=[self.app_layout])
        
    @default('project_list')
    def default_project_list(self):
        sseturn [('John', 0), ('Jamie', 1), ('Kay', 2)]
        
    def display(self):
        display(self.app_tab)