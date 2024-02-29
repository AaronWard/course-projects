//
//  ViewController.swift
//  EggTimer
//
//
//  Created by Aaron Ward.
//  Copyright © 2020 The App Brewery. All rights reserved.
//

import UIKit
import AVFoundation

class ViewController: UIViewController {
    
    @IBOutlet weak var bannerLabel: UILabel!
    @IBOutlet weak var progressBar: UIProgressView!
    var timer = Timer()
    let eggTime = ["Soft": 300,"Medium": 560,"Hard": 720]  // Minutes
    var totalTime: Float = 0.0
    var player: AVAudioPlayer?

    
    @IBAction func hardnessPressed(_ sender: UIButton) {
        self.bannerLabel.text = "How do you like your eggs?"

        // Shadow the image of the clicked egg for .2 seconds
        sender.alpha = 0.8
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2 ){
                sender.alpha = 1.0
        }
        
        
        timer.invalidate()
        totalTime = Float(eggTime[sender.currentTitle!]!)
        var secondsPassed: Float = 0
        self.progressBar.progress = 0.0
        
        // Loop though number of seconds needed and alarm when finished
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) {(Timer) in
            if secondsPassed < self.totalTime {
                secondsPassed += 1.0
                self.progressBar.progress = (Float(secondsPassed) / self.totalTime)
             }
            else{
                self.timer.invalidate()
                self.bannerLabel.text = "DONE!"
                self.playSound()
            }
         }
        
    }
    
    func showClickedHardness(sender: UIButton){
        // Shadow the image of the clicked egg for .2 seconds
        sender.alpha = 0.8
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2 ){
                sender.alpha = 1.0
        }
    }
    
    // Function for playing alarm sound
    func playSound() {
        guard let url = Bundle.main.url(forResource: "alarm_sound", withExtension: "mp3") else { return }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            player = try AVAudioPlayer(contentsOf: url, fileTypeHint: AVFileType.mp3.rawValue)

            guard let player = player else { return }
            player.play()
        } catch let error {
            print(error.localizedDescription)
        }
    }
    
    
}
