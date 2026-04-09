const App = {
    delimiters: ['[[', ']]'],
    data() {
        return {
            showMode:0,
            selectedFile: null,
            previewUrl: '',
            predictionResult: null,
            isUploading: false,
            isDragOver: false,
            errorMessage: '',
            resultHighlighted: false,
            isCameraOn: false,
            cameraError: '',
            cameraStream: null,
            apiUrl: '/predict/'
        };
    },
    computed: {
        topPredictions() {
            if (!this.predictionResult || !Array.isArray(this.predictionResult.all_predictions)) {
                return [];
            }

            return [...this.predictionResult.all_predictions]
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 3);
        }
    },
    methods: {
        scrollToWorkspace() {
            const el = document.getElementById('workspace');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        },
        scrollToCamera() {
            const el = document.getElementById('camera-card');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        },
        scrollToResultPanel() {
            const el = document.getElementById('result-panel');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        },
        triggerFilePicker() {
            if (this.$refs.fileInput) {
                this.$refs.fileInput.click();
            }
        },
        clearPreview() {
            if (this.previewUrl) {
                URL.revokeObjectURL(this.previewUrl);
                this.previewUrl = '';
            }
        },
        stopCamera() {
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => track.stop());
                this.cameraStream = null;
            }

            if (this.$refs.video) {
                this.$refs.video.srcObject = null;
            }

            this.isCameraOn = false;
        },
        async startCamera() {
            this.cameraError = '';

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment'
                    },
                    audio: false
                });

                this.stopCamera();
                this.cameraStream = stream;

                if (this.$refs.video) {
                    this.$refs.video.srcObject = stream;
                    this.isCameraOn = true;
                }
            } catch (error) {
                console.error(error);
                this.cameraError = 'Няма достъп до камерата или браузърът не позволява използването ѝ.';
            }
        },
        takePhoto() {
            this.cameraError = '';

            if (!this.isCameraOn || !this.cameraStream || !this.$refs.video || !this.$refs.canvas) {
                this.cameraError = 'Камерата не е активна.';
                return;
            }

            const video = this.$refs.video;
            const canvas = this.$refs.canvas;
            const context = canvas.getContext('2d');

            if (!video.videoWidth || !video.videoHeight) {
                this.cameraError = 'Камерата все още се инициализира. Опитайте отново след секунда.';
                return;
            }

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (!blob) {
                    this.cameraError = 'Неуспешно заснемане на снимката.';
                    return;
                }

                const file = new File([blob], `camera-${Date.now()}.jpg`, {
                    type: 'image/jpeg'
                });

                this.stopCamera();
                this.setSelectedFile(file);
            }, 'image/jpeg', 0.92);
        },
        resetForm() {
            this.selectedFile = null;
            this.predictionResult = null;
            this.isUploading = false;
            this.isDragOver = false;
            this.errorMessage = '';
            this.resultHighlighted = false;
            this.cameraError = '';
            this.stopCamera();
            this.clearPreview();

            if (this.$refs.fileInput) {
                this.$refs.fileInput.value = '';
            }
        },
        validateFile(file) {
            if (!file) {
                return 'Моля, изберете изображение.';
            }

            if (!file.type || !file.type.startsWith('image/')) {
                return 'Невалиден файл. Моля, изберете изображение (JPG, PNG, WEBP).';
            }

            return '';
        },
        setSelectedFile(file) {
            this.errorMessage = '';
            this.predictionResult = null;
            this.resultHighlighted = false;

            const validationError = this.validateFile(file);
            if (validationError) {
                this.resetForm();
                this.errorMessage = validationError;
                return;
            }

            this.clearPreview();
            this.selectedFile = file;
            this.previewUrl = URL.createObjectURL(file);
        },
        onFileChange(event) {
            const file = event.target.files[0] || null;
            this.setSelectedFile(file);
            this.isDragOver = false;
        },
        handleDrop(event) {
            this.isDragOver = false;
            const file = event.dataTransfer.files[0] || null;
            this.setSelectedFile(file);
        },
        async uploadImage() {
            if (this.isUploading) {
                return;
            }

            const validationError = this.validateFile(this.selectedFile);
            if (validationError) {
                this.errorMessage = validationError;
                return;
            }

            this.scrollToResultPanel();

            const formData = new FormData();
            formData.append('image', this.selectedFile);

            this.isUploading = true;
            this.errorMessage = '';
            this.predictionResult = null;
            this.resultHighlighted = false;

            try {
                const response = await axios.post(this.apiUrl, formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    },
                    timeout: 30000
                });

                if (response.data && response.data.success && response.data.result) {
                    this.predictionResult = response.data.result;

                    this.$nextTick(() => {
                        this.scrollToResultPanel();
                        this.resultHighlighted = true;

                        window.setTimeout(() => {
                            this.resultHighlighted = false;
                        }, 4000);
                    });
                } else {
                    this.errorMessage = 'Сървърът върна неочакван отговор.';
                }
            } catch (error) {
                if (error.response && error.response.data && error.response.data.error) {
                    this.errorMessage = error.response.data.error;
                } else if (error.code === 'ECONNABORTED') {
                    this.errorMessage = 'Обработката отне твърде дълго. Опитайте отново.';
                } else {
                    this.errorMessage = 'Възникна грешка при обработката на изображението.';
                }

                console.error(error);
            } finally {
                this.isUploading = false;
            }
        }
    },
    beforeUnmount() {
        this.stopCamera();
        this.clearPreview();
    }
};

Vue.createApp(App).mount('#app');
