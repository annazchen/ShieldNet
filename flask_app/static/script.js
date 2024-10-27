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
        console.log(file)
    });
    $("#csv_input_2").change(function(){
        if(document.getElementById("csv_input_2").files.length == 0 ){
            document.getElementById("check_csv_input").innerHTML= "No files selected";
        }else{
            var filename = $('#csv_input_2').val().split('\\').pop();
            document.getElementById("check_csv_input").innerHTML= filename + " selected";
            file = document.getElementById("csv_input_2").files[0]
        }
        console.log(file)
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
    console.log(file)
}
function dragOverHandler(ev) {
console.log("File(s) in drop zone");

// Prevent default behavior (Prevent file from being opened)
ev.preventDefault();
}
function loading(){
$("#logo-loading").show();
$("#logo").hide();
console.log(file)
}
// var data= []
$(document).on("click", ":submit", function (e) {
    if (document.getElementById("csv_input").files.length == 0){
        if (document.getElementById("csv_input_2").files.length == 0 ){
            e.preventDefault();
            document.getElementById("check_csv_input").innerHTML= "Please upload a file!";
        } 
    }else{
        loading();
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            alert('File uploaded successfully!');
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('Error uploading file.');
        });
    }
    console.log(file)
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
