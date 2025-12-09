# ☁️ Google Colab Training Guide (Zero-Error Method)

Follow these steps exactly to train your model on Google's Free GPU.

## Phase 1: Preparation (On your PC)
1.  **Zip your Project**:
    *   Go to your `LeafModel` folder.
    *   Select `src`, `configs`, `requirements.txt`, and your **`data`** folder (The Raw one!).
    *   *Note: Do NOT zip `data_processed`. The script will recreate it in Colab.*
    *   **Right Click > Send to > Compressed (zipped) folder**.
    *   Name it `project_upload.zip`. (This makes uploading 10,000 files much faster).

## Phase 2: Setup Colab
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Click **"Upload"** tab and upload your `Run_in_Colab.ipynb` file.
3.  **Enable GPU**:
    *   Go to Top Menu: **Runtime** > **Change runtime type**.
    *   Select **T4 GPU** (or any GPU available).
    *   Click **Save**.

## Phase 3: Upload Data
### Option A: Direct Upload (Best for < 1GB)
1.  On the Left Sidebar, click the **Folder Icon 📁**.
2.  Drag and drop your `project_upload.zip`.

### Option B: Google Drive (Required for 6GB!)
*Direct upload of 6GB will likely fail or take 1+ hour.*
1.  Upload `project_upload.zip` to your **Google Drive** first (Search "Google Drive" in Google).
2.  In Colab, click the **Folder Icon** > **Mount Drive Icon** (folder with Drive logo).
3.  Grant permission.
4.  Copy your zip from `drive/MyDrive/...` to the Colab workspace:
    ```python
    !cp "/content/drive/MyDrive/project_upload.zip" .
    ```


## Phase 4: Run Training
1.  **Unzip**: Run the first cell I added (Step 0) to unzip your files automatically.
2.  **Install**: Run Step 1 to install dependencies.
3.  **Organize**: Run Step 2.1 to organize your data.
    *   *Note*: If it says "Already exists", that's fine.
4.  **Train**: Run Step 3.
    *   Watch the `Epoch` progress.
    *   Wait for "Exporting to TFLite..." success message.

## Phase 5: Download (The Fruit of your Labor)
1.  The last cell will trigger a download for `model.tflite`.
2.  If chrome blocks it, check the `exports` folder in the sidebar and Right Click > Download manually.

---

## ⚠️ Common "Gotchas" (Pro Tips)
*   **"FileNotFoundError"**: You probably didn't upload the zip, or you are in the wrong folder. Run `!ls` to check where you are.
*   **"Disconnecting..."**: Colab kills the session if you close the tab. Keep the tab open!
*   **"Out of Memory"**: If this happens, open `configs/config.yaml` and change `batch_size: 32` to `16`.
