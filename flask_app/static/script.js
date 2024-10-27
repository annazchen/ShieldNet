var file = new File([""], "filename");
$(document).ready(function(){
    $("#csv_input").change(function(){
        if(document.getElementById("csv_input").files.length == 0 ){
            document.getElementById("check_csv_input").innerHTML= "No files selected";
        }else{
            var filename = $('#csv_input').val().split('\\').pop();
            document.getElementById("check_csv_input").innerHTML= filename + " selected";
            file = document.getElementById("csv_input").files[0]
        }
        console.log("ther", file)
    });
    $("#csv_input_2").change(function(){
        if(document.getElementById("csv_input_2").files.length == 0 ){
            document.getElementById("check_csv_input").innerHTML= "No files selected";
        }else{
            var filename = $('#csv_input_2').val().split('\\').pop();
            document.getElementById("check_csv_input").innerHTML= filename + " selected";
            file = document.getElementById("csv_input_2").files[0]
        }
        console.log("or here", file)
    });
    
});

function dropHandler(ev) {
    console.log("File(s) dropped");
  
    ev.preventDefault();
  
    if (ev.dataTransfer.items) {
        item = ev.dataTransfer.items[0];
      // Access the files
      if (item.kind === "file") {
        const temp_file = item.getAsFile();
        document.getElementById("check_csv_input").innerHTML= temp_file.name +" selected";
        file = temp_file
      }
    }
    console.log("PELASEFDS", file)
}
// function dragOverHandler(ev) {
// console.log("File(s) in drop zone");

// // Prevent  behavior (Prevent file from being opened)
// ev.preventDedefaultfault();
// }
function loading(){
    $("#logo-loading").show();
    $("#logo").hide();
    console.log("HERER", file)
}

$(document).on("click", ":submit", function (e) {
    e.preventDefault()
    if (document.getElementById("csv_input").files.length == 0){
        if (document.getElementById("csv_input_2").files.length == 0 ){
            document.getElementById("check_csv_input").innerHTML= "Please upload a file!";
        } 
    }else{
        loading();
        const formData = new FormData();
        formData.append('csv_file', file);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            console.log('Success:', data);
            window.location.href = "http://localhost:5000/display"
            alert('File uploaded successfully!');
        })
        .catch((error) => {
            console.log('Error:', error);
            console.error('Error:', error);
            alert('Error uploading file.');
        });
    }
    console.log("goT you fam", file)
});

// async function fetchCSV(url) {
//     try {
//         const response = await fetch(url);
//         const data = await response.text();
//         document.getElementById('output').innerText = data;
//     } catch (error) {
//         console.error('Error fetching CSV:', error);
//     }
//}
