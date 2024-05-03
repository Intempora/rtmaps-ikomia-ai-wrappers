from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_yolop_v2", auto_connect=True)

# Run on your image  
wf.run_on(url="https://www.cnet.com/a/img/resize/4797a22dd672697529df19c2658364a85e0f9eb4/hub/2023/02/14/9406d927-a754-4fa9-8251-8b1ccd010d5a/ring-car-cam-2023-02-14-14h09m20s720.png?auto=webp&width=1920")

# Inpect your result
display(algo.get_image_with_graphics())
display(algo.get_output(0).get_overlay_mask())