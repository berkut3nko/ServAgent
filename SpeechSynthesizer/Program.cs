using System;
using System.Speech.Synthesis;

// Initialize a new instance of the SpeechSynthesizer.
SpeechSynthesizer synth = new SpeechSynthesizer();

foreach (var voice in synth.GetInstalledVoices())
{
    Console.WriteLine(voice.VoiceInfo.Description);
}

// Configure the audio output.
synth.SetOutputToDefaultAudioDevice();

// Speak a string.
synth.Speak("Цей приклад демострує приклад використання синтезатора");

Console.WriteLine();
Console.WriteLine("Press any key to exit...");
Console.ReadKey();