const App = {
    delimiters: ['[[', ']]'], // Променяме синтаксиса на [[ ]]
    data() {
        return {
            showMode: 0,
            selectedFile: null,
            predictionResult: null,
            isUploading: false,
            errorMessage: ''
        }
    },
    methods: {
        onFileChange(event) {
            this.selectedFile = event.target.files[0] || null;
            this.errorMessage = '';
        },
        async uploadImage() {
            if (!this.selectedFile) {
                this.errorMessage = 'Моля, изберете изображение.';
                return;
            }

            const formData = new FormData();
            formData.append('image', this.selectedFile);

            this.isUploading = true;
            this.errorMessage = '';
            this.predictionResult = null;

            try {
                const response = await axios.post('/predict/', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data'
                    }
                });

                this.predictionResult = response.data.result;
            } catch (error) {
                this.errorMessage = 'Възникна грешка при обработката на изображението.';
                console.error(error);
            } finally {
                this.isUploading = false;
            }
        }
    },
    created: function(){
        console.log("created *******************");
        this.showMode = 0;
    }
}

Vue.createApp(App).mount('#app')
