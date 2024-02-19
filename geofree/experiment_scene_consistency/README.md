# Scene Style Consistency:

This readme provides an overview of the modifications aimed at addressing the issue of style consistency between consecutive synthesized frames.

For instance, consider the following gif extracted from one of the demos presented in the paper, illustrating a significant shift in scene style:

![Visualization of scene style shift](https://drive.google.com/uc?export=view&id=1V94b4Nd-fVi4Trd_7GZ5SR6hoQuAcLlS)


Within this folder, three different ideas are explored:

a) **Local Style Transfer**: This involves artistic style transfer for specific objects in the scene. Related files include:
   - `segmentation.py`: An interface for the SegFormer pretrained model.
   - `styletransfer.py`: A simple implementation of the Neural Style transfer paper.
   - `braidance.py`: An updated version to run all steps in an interactive window (synthesizing, segmenting, object selection, and style transfer). Note that this `braidance.py` should replace the one in the scripts directory.
   
   Output should be:
   ![Sample output](https://drive.google.com/uc?id=1DWA7FkOvdRLdqonZjD2fGwMRSi6J9hcd)


b) **Scene Style Encoding**: Two different approaches were attempted:
   - The primary approach involves learning the scene codes as described in the report. Related files include:
     - `image_vqgan_encoding.ipynb`: Pre-encoding the entire dataset of 400 scenes before training for reduced memory resource usage.
     - `learnable_style.ipynb`: Actual experimentation of learning the style described in the report.
   - The second approach, which proved unsuccessful and therefore not mentioned in the report, aimed to extract the scene style using an aggregation function of all image codes. The notebook `non_learned_scene_code.ipynb` contains numerous experiments related to this idea, albeit somewhat messy. Apologies in advance for the lack of organization.


